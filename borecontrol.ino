#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <ESP_EEPROM.h>
#include <Wire.h>
#include <I2C_RTC.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <Adafruit_ADS1X15.h>

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64

/* Put your SSID & Password */
const char* ssid = "nodemcu-test";          // Enter SSID here
const char* password = "nodemcu-test";  //Enter Password here

/* Put IP Address details */
IPAddress local_ip(192, 168, 1, 1);
IPAddress gateway(192, 168, 1, 1);
IPAddress subnet(255, 255, 255, 0);

static DS3231 RTC;
ESP8266WebServer server(80);

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);
Adafruit_ADS1115 ads;  /* Use this for the 16-bit version */

uint8_t pinPool[10] = {D0, D1,D2, D3, D4, D5, D6, D7, D8, D9};
uint8_t LED1pin = D0;
uint8_t LED2pin = D3;
uint8_t LED3pin = D4;
uint8_t LED4pin = D5;
uint8_t LED5pin = D6;
uint8_t LED6pin = D7;
uint8_t LED7pin = D8;
uint8_t LED8pin = D9;

bool LEDstatus[] = { false, false, false, false, false, false, false, false };
int ledOpFlag = 1;
unsigned int endStopTimer[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
unsigned long timeNow = 0;
unsigned long timeLast = 0;
unsigned int minutes = 0;
uint8_t seconds = 0;
uint8_t mock_Hr = 0; // when RTC not present, mock_Hr is used to set the time
uint8_t mock_Min = 0; //when RTC not present, mock_Min is used to set the time

bool isLedSkippedAuto[] = { false, false, false, false, false, false, false, false }; //for skipping interval & RTC timer when this flag is true, not writing to EEPROM
bool justTriggerForLatch[] = { false, false, false, false, false, false, false, false }; //for skipping interval & RTC timer when this flag is true, not writing to EEPROM
String received_cnf_res = "\n";
bool serialPrintEnable=false;
bool rtcDetected=false;
bool adsDetected=false;
float adc0;
volatile u_int64_t pulseCounterVal[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
struct GpioInputMap {
  bool is_pullup_pulldown=true;
  int8_t pin=-1;
  // uint8_t trigger_pin;
};
struct MyEEPROMStruct {
  uint8_t ledonHr[8];
  uint8_t ledonMin[8];
  uint8_t ledoffHr[8];
  uint8_t ledoffMin[8];
  int ledInterval_onMin[8];
  int ledInterval_offMin[8];
  bool ledCouple[4];
  bool isLedTimer[8];
  bool isLedInterval[8];
  bool isLedRtcInterval[8];
  bool pulseCounterInputPinMap[8];
  char colorCode[7]; //+1 for '\0'
  char lcdADDR[3]; //+1 for '\0'
  char ledSquence[9]; //+1 for '\0'
  char displayingPanels[9]; //+1 for '\0' // 11111111 - all panels are displayed, 00000000 - no panel is displayed
  uint8_t currentSensorrelatedIOs; //34 -> if op 3 makes the OP on, op 4 if the current exceeeds & its emergency stop
  float currentSensorThreshold; //x value, upper value is over current value
  bool overCurrentDetected;
  bool showUiDefault=false;
  GpioInputMap gpio_input_map[8]; 
} eepromVar1, eepromVar2;  //TODO eepromVar2 size is too big for use case

void gpioOn(uint8_t pinNo) {
  // Serial.println("GPIO HIGH = "+String(pinNo));
  // digitalWrite(pinNo, HIGH);
  digitalWrite(pinNo, LOW);
}
void gpioOff(uint8_t pinNo) {
  // Serial.println("GPIO LOW = "+String(pinNo));
  // digitalWrite(pinNo, LOW);
  digitalWrite(pinNo, HIGH);
}

void allGpioOff() {
  Serial.println("allGpioOff from SETUP");
  uint8_t GPIO_pins[] = { LED1pin,
                          LED2pin,
                          LED3pin,
                          LED4pin,
                          LED5pin,
                          LED6pin,
                          LED7pin,
                          LED8pin };
  for (uint8_t i = 0; i < 8; i++) {
    gpioOff(GPIO_pins[i]);
    // if (eepromVar2.isLedInterval[i]) {  //TODO: couple condion is remaining //TODO; confusing
    //   //turn on the led 1
    //   LEDstatus[i] = true;
    // }
  }
}

void allGpioOn() {
  uint8_t GPIO_pins[] = { LED1pin,
                          LED2pin,
                          LED3pin,
                          LED4pin,
                          LED5pin,
                          LED6pin,
                          LED7pin,
                          LED8pin };
  for (uint8_t i = 0; i < 8; i++) {
    gpioOn(GPIO_pins[i]);
  }
}

String enforce_4_digit_string(long valstr){
  //if string is 9999 - its asumed to display 9999 seconds
  //if string is greater than 9999 - its asumed to display 9999 seconds in hour or minutes
  //lets convert sends into minutes, if its larger than 59 minutes, then lets display XH+(X hours +)
  //if XH+ value is larger than 99H+ then lets show 99H++.
  if (valstr <= 9999) {
    char buf[5];
    sprintf(buf, "%04ld", valstr);
    return String(buf);
  }
  
  long totalMinutes = valstr / 60;
  if (totalMinutes <= 9) {
    return String(totalMinutes) + "m++";
  } else if (totalMinutes <= 59) {
    return String(totalMinutes) + "m+";
  }
  
  long totalHours = totalMinutes / 60;
  if (totalHours < 10) {
    return String(totalHours) + "h++";
  } if (totalHours >= 10 && totalHours < 100) {
    return String(totalHours) + "h+";
  } else if (totalHours >= 100) {
    return "99h+";
  }
  return String(valstr);
}

void displayLedStatus() {
  String timeDisplayVal;
  if (RTC.isConnected() && RTC.isRunning()) {
    timeDisplayVal=String(RTC.getDay()) + "-" + String(RTC.getMonth()) + "-" + String(RTC.getYear()) + " " + String(RTC.getHours()) + ":" + String(RTC.getMinutes()) + ":" + String(RTC.getSeconds()) + " &nbsp Running: &nbsp" + String(minutes)+"\t::\t"; 
    Serial.print(timeDisplayVal);
  }else {
    timeDisplayVal=""+String(mock_Hr)+"."+String(mock_Min);
    Serial.print(timeDisplayVal+String("\t::\t"));
  }
  for (uint8_t i = 0; i < 8; i++) {
    Serial.print(String(LEDstatus[i]) + "\t");
  }
  Serial.println();

  String contents[8];
  char runningMinutes[7];
  sprintf(runningMinutes, "%6d", minutes);

  for (uint8_t i = 0, ledCoupleCount=0; i < 8; ++i, ledCoupleCount=i/2) {
    //TIME=23:59 ::Name=1 ::Status=0 ::isRTC=1 ::isInterval=1 ::isCouple=0 ::isRTCIterval=0 ::RTCCNF=12.32-12.45 ::INTCNF=9999-5545 ::REM=9999-5545 ::EndStop=12 
    //T=time, N=Name, D0=digital_pin_number H=high/low R=isRTC? I=isInterval C=isCouple RI=isRTCInterval RE=RTC eeprom IE=Iterval EEprom -I remaining intraval time -T remaining time to simple off
    // String contentData=String("T="+timeDisplayVal+" N=LED"+(i+1)+ " H=" + LEDstatus[i]+ " R=" +eepromVar1.isLedTimer[i] + " I=" + eepromVar1.isLedInterval[i]+ + " C=" + eepromVar1.ledCouple[ledCoupleCount] + " RI=" + eepromVar1.isLedRtcInterval[i] + " RE=" + eepromVar1.ledonHr[i]+"."+eepromVar1.ledonMin[i]+"-"+eepromVar1.ledoffHr[i]+"."+eepromVar1.ledoffMin[i] + " IE" + eepromVar1.ledInterval_onMin[i]+"-" + eepromVar1.ledInterval_offMin[i] + " -I"+ eepromVar2.ledInterval_onMin[i]+ "-" + eepromVar2.ledInterval_offMin[i]+" -T"+endStopTimer[i]+"     ");

    char hrBuf[3], minBuf[3], offHrBuf[3], offMinBuf[3];
    sprintf(hrBuf, "%02d", eepromVar1.ledonHr[i]);
    sprintf(minBuf, "%02d", eepromVar1.ledonMin[i]);
    sprintf(offHrBuf, "%02d", eepromVar1.ledoffHr[i]);
    sprintf(offMinBuf, "%02d", eepromVar1.ledoffMin[i]);

    String contentData = 
      "T=" + String(timeDisplayVal) +
      " N=LED" + String(i+1) +
      " D" + String(eepromVar1.ledSquence[i]) +
      " H=" + String(LEDstatus[i]) +
      " R=" + String(eepromVar1.isLedTimer[i]) +
      " I=" + String(eepromVar1.isLedInterval[i]) +
      " C=" + String(eepromVar1.ledCouple[ledCoupleCount]) +
      " RI=" + String(eepromVar1.isLedRtcInterval[i]) +
      " RE=" + String(hrBuf) + "." + String(minBuf) + "-" + String(offHrBuf) + "." + String(offMinBuf) +
      " IE=" + enforce_4_digit_string(eepromVar1.ledInterval_onMin[i]) + "-" + enforce_4_digit_string(eepromVar1.ledInterval_offMin[i]) +
      " -I=" + enforce_4_digit_string(eepromVar2.ledInterval_onMin[i]) + "-" + enforce_4_digit_string(eepromVar2.ledInterval_offMin[i]) +
      " RNG=" + String(runningMinutes) +
      " -T=" + String(endStopTimer[i]) + "     ";
      if(adsDetected && (LEDstatus[eepromVar1.currentSensorrelatedIOs/10-1] || LEDstatus[eepromVar1.currentSensorrelatedIOs%10-1])){
        contentData += " ADC0=" + String(adc0)+ "      ";
      }
    
    Serial.println(contentData);
    contents[i]=contentData;
  }
  Serial.println("---- -------------------------------------------- After 3 seconds -----");
  displayOLEDscreen(contents);
}

void writeToGpio() {
  uint8_t GPIO_pins[] = { LED1pin,
                          LED2pin,
                          LED3pin,
                          LED4pin,
                          LED5pin,
                          LED6pin,
                          LED7pin,
                          LED8pin };
  for (uint8_t i = 0; i < 8; i++) {
    if (LEDstatus[i] && !isLedSkippedAuto[i]) {
      gpioOn(GPIO_pins[i]);
    }
    if (!LEDstatus[i] && !isLedSkippedAuto[i]) {
      gpioOff(GPIO_pins[i]);
    }
    if(i%2==0 && eepromVar1.ledCouple[i/2] && LEDstatus[i] && !justTriggerForLatch[i]){ //if coupled partner1 is ON for 3 sec then check the latch flag, clear the next partner's latch flag
      justTriggerForLatch[i]=true;
      justTriggerForLatch[i+1]=false;
      continue;
    }
    if(i%2==0 && eepromVar1.ledCouple[i/2] && LEDstatus[i] && justTriggerForLatch[i]){ //if latch flag is set then, off the LED (not status, only supply) so that ON contactor gets latched & supply cut for input
      gpioOff(GPIO_pins[i]); //may the contactor latch as its coupled contactor
    }
    if(i%2==1 && eepromVar1.ledCouple[i/2] && LEDstatus[i] && !justTriggerForLatch[i]){//if coupled partner2 is ON for 3 sec then check the latch flag, clear the prev partner's latch flag
      justTriggerForLatch[i]=true;
      justTriggerForLatch[i-1]=false;
      continue;
    }
    if(i%2==1 && eepromVar1.ledCouple[i/2] && LEDstatus[i] && justTriggerForLatch[i]){//if latch flag is set then, off the LED (not status, only supply) so that ON contactor gets latched & supply cut for input
      gpioOff(GPIO_pins[i]); //may the contactor latch as its coupled contactor
    }
  }
}

void setInitEEPROM() {
  for(uint8_t i=0, ledCoupleCount=0; i<8 ; ++i, ledCoupleCount=i/2){
    eepromVar1.isLedTimer[i] = false;
    eepromVar1.isLedInterval[i] = false;
    eepromVar1.ledonHr[i] = 18;
    eepromVar1.ledoffHr[i] = 23;
    eepromVar1.ledonMin[i] = 59;
    eepromVar1.ledoffMin[i] = 59;
    eepromVar1.ledInterval_onMin[i] = 120;
    eepromVar1.ledInterval_offMin[i] = 3600;
    eepromVar1.ledCouple[ledCoupleCount]=false;
    eepromVar1.pulseCounterInputPinMap[i]=false;
    eepromVar1.gpio_input_map[i].pin=-1;
    // eepromVar1.gpio_input_map[i].trigger_pin=-1;
    eepromVar1.gpio_input_map[i].is_pullup_pulldown=true;
  }
  strcpy(eepromVar1.lcdADDR, "3C");
  strcpy(eepromVar1.ledSquence, "03456789"); //1 & 2 are for I2C
  strcpy(eepromVar1.displayingPanels, "11111111"); //all panels should show by default
  eepromVar1.currentSensorrelatedIOs = 34; //op 3 - makes contactor on; 4 - makes contactor OFF ; starts from 1
  eepromVar1.currentSensorThreshold=2.2;//over 2.2 voltage detcion, high current is followed.
  eepromVar1.overCurrentDetected=false;
  eepromVar2 = eepromVar1;
}

void checkEEPROM_percentage() {
  if (EEPROM.percentUsed() >= 0) {
    EEPROM.get(0, eepromVar1);
    eepromVar2 = eepromVar1;
    // eepromVar1.anInteger++;     // make a change to our copy of the EEPROM data
  for(uint8_t i=0; i<8 ; i++){
    Serial.println(String(eepromVar1.isLedTimer[i]) + "::" + String(eepromVar1.isLedInterval[i]) + "::" + String(eepromVar1.ledonHr[i]) + "." + String(eepromVar1.ledonMin[i]) + "-" + String(eepromVar1.ledoffHr[i]) + "." + String(eepromVar1.ledoffMin[i]) + "::" + String(eepromVar1.ledInterval_onMin[i]) + "-" + String(eepromVar1.ledInterval_offMin[i]));
  }
    Serial.println("EEPROM has data from a previous run.");
    Serial.print(EEPROM.percentUsed());
    Serial.println("% of ESP flash space currently used");
  } else {
    Serial.println("EEPROM size changed - EEPROM data zeroed - commit() to make permanent");
  }
}

template <typename T>
void appendHtml(String& output, const T& value) {
  output += value;
}

template <typename T, typename... Values>
void appendHtml(String& output, const T& value, const Values&... values) {
  output += value;
  appendHtml(output, values...);
}

String buildLedHtml(u_int8_t ledNoInt, bool coupleFlag = false, bool rtcIntFlag = false) { 
  String ledNo = String(ledNoInt+1); //ledNo is display number so starting from 1
  const char* divId = "<div id=\"";
  const char* tag_ = "\">";
  const char* It = "<input ";
  const char* IV = "value=\"System";
  const char* tag1_ = "\"/>";
  const char* IR2 = " checked ";
  const char* IR = "type=\"radio\" name=\"led";
  const char* Os = "Options\" ";
  const char* Id = "id=\"led";
  String ptr = "";
  const char* v = "value=\"l";
  const char* o = "o\"";
  const char* tag2_ = "/>";
  const char* Lo = "<label for=\"led";
  const char* Lo2 = "_off\">OFF";
  const char* off = "_off\" ";
  const char* LE = "</label>";
  const char* RT = "_rtc\" ";
  const char* R = "r\"";
  const char* Lo3 = "_rtc\">Rtc time";
  const char* IL = "_interval\" ";
  const char* IN = "_input\" ";
  const char* CN = "_counter\" ";
  const char* I = "i\"";
  const char* IP = "ip\"";
  const char* C = "c\"";
  const char* Lo4 = "_interval\">Interval";
  const char* Lo5 = "_input\">Input";
  const char* Lo6 = "_counter\">Counter";
  const char* BI = "<button id=\"";
  const char* MID = "m\">";
  String AN = "<a id='l" + ledNo + "a' href='/led";
  const char* ANEON = "on'>";
  const char* ANEOFF = "off'>";
  const char* MO = "ON";
  const char* MOF = "OFF";
  const char* BE = "</button>";
  const char* IR3 = "type=\"text\" name=\"l";
  const char* S = " ";
  const char* Id2 = "id=\"l";
  const char* dE = "</div>";
  const char* sp= "<span>";
  const char* sp_= "</span>";

  bool timerFlag = eepromVar2.isLedTimer[ledNoInt];
  bool intervalFlag = eepromVar2.isLedInterval[ledNoInt];
  bool inputGpioFlag = eepromVar2.gpio_input_map[ledNoInt].pin!=-1;
  bool inputPulseCounterFlag = eepromVar2.pulseCounterInputPinMap[ledNoInt];

  appendHtml(ptr, divId, "l", ledNo, tag_, "\n");
  appendHtml(ptr, It, IV, ledNo, tag1_, "\n");
  appendHtml(ptr, divId, "\"class=\"rdiv", tag1_, "\n");
  if (!(timerFlag || intervalFlag)) {  //off radio check
    appendHtml(ptr, sp, It, IR2, IR, ledNo, Os, Id, ledNo, off, v, ledNo, o, tag2_, "\n");
  } else {
    appendHtml(ptr, sp, It, IR, ledNo, Os, Id, ledNo, off, v, ledNo, o, tag2_, "\n");
  }
  appendHtml(ptr, Lo, ledNo, Lo2, LE, sp_, "\n");
  if (timerFlag) {  //timer radio check
    appendHtml(ptr, sp, It, IR2, IR, ledNo, Os, Id, ledNo, RT, v, ledNo, R, tag2_, "\n");
  } else {
    appendHtml(ptr, sp, It, IR, ledNo, Os, Id, ledNo, RT, v, ledNo, R, tag2_, "\n");
  }
  appendHtml(ptr, Lo, ledNo, Lo3, LE, sp_, "\n");
  if (intervalFlag) {  //interval radio check
    appendHtml(ptr, sp, It, IR2, IR, ledNo, Os, Id, ledNo, IL, v, ledNo, I, tag2_, "\n");
  } else {
    appendHtml(ptr, sp, It, IR, ledNo, Os, Id, ledNo, IL, v, ledNo, I, tag2_, "\n");
  }
  appendHtml(ptr, Lo, ledNo, Lo4, LE, sp_, "\n");
  if (inputGpioFlag) {  //Input radio check
    appendHtml(ptr, sp, It, IR2, IR, ledNo, Os, Id, ledNo, IN, v, ledNo, IP, tag2_, "\n");
  } else {
    appendHtml(ptr, sp, It, IR, ledNo, Os, Id, ledNo, IN, v, ledNo, IP, tag2_, "\n");
  }
  appendHtml(ptr, Lo, ledNo, Lo5, LE, sp_, "\n");
  if (inputPulseCounterFlag) {  //Input counter radio check
    appendHtml(ptr, sp, It, IR2, IR, ledNo, Os, Id, ledNo, CN, v, ledNo, C, tag2_, "\n");
  } else {
    appendHtml(ptr, sp, It, IR, ledNo, Os, Id, ledNo, CN, v, ledNo, C, tag2_, "\n");
  }
  appendHtml(ptr, Lo, ledNo, Lo6, LE, sp_, "\n");
  appendHtml(ptr, dE, "\n");

  if (LEDstatus[ledNo.toInt() - 1] == false) {  //manual btn on off
    appendHtml(ptr, AN, ledNo, ANEON, BI, "l", ledNo, MID, MO, BE, "</a>\n");
  } else {
    appendHtml(ptr, AN, ledNo, ANEOFF, BI, "l", ledNo, MID, MOF, BE, "</a>\n");
  }

  //input placeholder code
  if (ledNoInt>=0 && ledNoInt<8) {
    if (endStopTimer[ledNoInt] > 0) {
      appendHtml(ptr, It, IR3, ledNo, R, S, Id2, ledNo, R, " placeholder='", String(endStopTimer[ledNoInt]), "min - Endstop remaining' disabled ", tag2_, "\n");
    } else if ((ledNoInt>0 && ledNoInt%2==1) && coupleFlag) {
      appendHtml(ptr, It, IR3, ledNo, R, S, Id2, ledNo, R, " placeholder='Coupled: Dont set time' ", tag2_, "\n");
    } else if (eepromVar2.isLedInterval[ledNoInt]) {
      appendHtml(ptr, It, IR3, ledNo, R, S, Id2, ledNo, R, " placeholder='", String(eepromVar2.ledInterval_onMin[ledNoInt]), "-", String(eepromVar2.ledInterval_offMin[ledNoInt]), "' ", tag2_, "\n");
    } else if (eepromVar2.gpio_input_map[ledNoInt].pin!=-1){
      appendHtml(ptr, It, IR3, ledNo, R, S, Id2, ledNo, R, " placeholder='#", String(eepromVar2.gpio_input_map[ledNoInt].is_pullup_pulldown ? '1' : '0'), "-", String(eepromVar2.gpio_input_map[ledNoInt].pin), "' ", tag2_, "\n");
    } else if (eepromVar2.pulseCounterInputPinMap[ledNoInt]){
      appendHtml(ptr, It, IR3, ledNo, R, S, Id2, ledNo, R, " placeholder='++", String(pulseCounterVal[ledNoInt]), "' ", tag2_, "\n");
    } else {
      appendHtml(ptr, It, IR3, ledNo, R, S, Id2, ledNo, R, " placeholder='", String(eepromVar2.ledonHr[ledNoInt]), ".", String(eepromVar2.ledonMin[ledNoInt]), "-", String(eepromVar2.ledoffHr[ledNoInt]), ".", String(eepromVar2.ledoffMin[ledNoInt]), "' ", tag2_, "\n");
    }
  } else {
    appendHtml(ptr, It, IR3, ledNo, R, S, Id2, ledNo, R, tag2_, "\n");
  }

  if ((ledNo.toInt()) % 2 != 0) {
    appendHtml(ptr, It, "id='l", ledNo, "cp' ", "type='checkbox' name='ledcouple' ");
    if (coupleFlag) { appendHtml(ptr, "checked"); }
    appendHtml(ptr, "/>Coupled");
  }
  appendHtml(ptr, "<input id='l", ledNo, "rtccp' type='checkbox' name='ledRTCI'");
  if (rtcIntFlag) { appendHtml(ptr, "checked"); }
  appendHtml(ptr, "/>use RTC range");

  if (isLedSkippedAuto[ledNo.toInt() - 1] == false) {
    appendHtml(ptr, "<a id='l", ledNo, "s' href='/ledskipauto?led=", ledNo, "'><button id='l", ledNo, "skip' class='skipB'>skip auto</button></a>\n");
  } else {
    appendHtml(ptr, "<a id='l", ledNo, "s' href='/ledunskipauto?led=", ledNo, "'><button id='l", ledNo, "skip' class='skipB skippedB'>skipped</button></a>\n");
  }

  if (LEDstatus[ledNo.toInt() - 1] == true) {
    appendHtml(ptr, "<p style=\"font-weight:bold;\">Currenly HIGH</p>");
  }

  appendHtml(ptr, dE, "\n");
  return ptr;
}

String renderTable(bool withFullUI = false) {
  String ptr = withFullUI ? "" : "<!DOCTYPE html>\n<html>\n<body>\n";
  if(!withFullUI){
    ptr += "<button id=\"fullui\">UI</button>\n";
    if (rtcDetected) {
      ptr += "&#x23F2; " + String(RTC.getDay()) + "-" + String(RTC.getMonth()) + "-" + String(RTC.getYear()) + " " + String(RTC.getHours()) + ":" + String(RTC.getMinutes()) + ":" + String(RTC.getSeconds()) + " &nbsp Running: &nbsp" + String(minutes);
    } else {
      ptr += "&#x23F2; MOCK: " + String(mock_Hr) + ":" + String(mock_Min) + " &nbsp Running: &nbsp" + String(minutes);
    }
    ptr += "<button id=\"timeset\">update TIME</button>\n";
  }

  ptr += "<table border=\"1\"><tr><th>Name</th><th>Show?</th><th>Hardware Pin</th><th>H/L</th><th>RTC</th><th>Interval</th><th>Input GPIO</th><th>RTC-cnf</th><th>Intervalcnf</th><th>Rem Interval</th><th>pulse counter</th><th>Coupled</th><th>Use RTC interval</th></tr>\n";
  for(uint8_t i=0,ledCoupleCount=0; i<8 ; ++i, ledCoupleCount=i/2){
    ptr += "<tr><td>led "+String(i+1)+"</td><td>"+String(eepromVar2.displayingPanels[i] == '1' ? "Yes" : "No")+"</td><td>" + String(eepromVar1.ledSquence[i]) +"</td><td style='background-color: "+(LEDstatus[i] ?"green":"red")+"; color: white;'>"+String(LEDstatus[i]) +"</td><td>" + String(eepromVar2.isLedTimer[i]) + "</td><td>" + String(eepromVar2.isLedInterval[i]) + "</td><td>" + "#"+String(eepromVar2.gpio_input_map[i].is_pullup_pulldown? '1':'0')+"-"+String(eepromVar2.gpio_input_map[i].pin) + "</td><td>" + String(eepromVar1.ledonHr[i]) + "." + String(eepromVar1.ledonMin[i]) + "-" + String(eepromVar1.ledo
  }
  ptr += "</table><div id=\"cnf\">"+received_cnf_res+"</div>";
  ptr += "<div id=\"current-sensor\">"+
    String("adsDetected=")+
    String(adsDetected)+
    String(" reading=")+
    String(adc0)+
    String(" sensorIO=")+
    String(eepromVar1.currentSensorrelatedIOs)+
    String(" current sensorthreshold=")+
    String(eepromVar1.currentSensorThreshold)+
    String(" current overCurrentDetected=")+
    String(eepromVar1.overCurrentDetected)+
    +"</div>";
  ptr += withFullUI ? "" : "\n<script>\nvar GI=e=>document.getElementById(e);\nUIBtn=GI('fullui');\nTS=GI('timeset');\nUIBtn.addEventListener('click',()=>window.location.href='/?ui=1');\nTS.addEventListener('click',()=>{var e=(new Date).toString().split(' '),t=e[0]+' '+e[1]+' '+e[2]+' '+e[4]+' '+e[3],l=confirm(JSON.stringify(t));const c=new URLSearchParams({date:t}).toString();l&&fetch(`/updateTime?${c}`)});\nsetTimeout(()=>{if (!(window.location.pathname == \"/\")) window.location.href = \"/\";},2e3);\n</script>\n</body>\n</html>\n";
  ptr += "<hr /><div>/eepromSet?coupled=0000&rtcinterval=00000000&l1r=01.59-23.59&l2r=01.59-23.59&l3r=01.59-23.59&l4r=01.59-23.59&l5r=100-3600&l6r=100-3600&l7r=100-3600&l8r=100-3600;;currentSensorIO=34;;currentSensorMax=3.2&l1=4&l2=0&l3=3&l4=5&l5=6&l6=7&l7=8&l8=9</div>";
  ptr += "<hr /><div>/eepromSet?l1r=#0-0&l2r=#1-1 :- if l1 low then trigger 0th output / if l2 is high then trigger 1st output";
  ptr += "<hr /><div>/led1on;;led1off</div>";
  ptr += "<hr /><div>/ledskipauto?led=1;;/ledunskipauto?led=1</div>";
  ptr += "<hr /><div>/serial</div>";
  return ptr;
}

String SendHTML() {

  //reusables
  String dE = "</div>";
  String BI = "<button id=\"";
  String BE = "</button>";
  String UP = "update";
  String tag_ = "\">";
  String colorCode = String(eepromVar1.colorCode).length() == 7 ? String(eepromVar1.colorCode) : "blueviolet";

  String ptr = "<!DOCTYPE html>\n<html>\n<head>\n<style>\n";
  ptr += "body{text-align:center;font-size:1.2rem;}\n";
  ptr += ".tmdn{visibility:hidden;}\n";
  ptr += ".rdiv{display:flex;flex-wrap: wrap;flex-direction: column;align-items: flex-start;height: 6em;}\n";
  ptr += "#l1,#l2,#l3,#l4,#l5,#l6,#l7,#l8{background-color:" + colorCode + ";display:flex;flex-direction:column;width:40%;padding:10px;margin:10px;text-align:center;box-shadow: 1px 1px 6px 2px #00000040;border-radius: 10px;}\n";
  ptr += "#l1m,#l2m,#l3m,#l4m,#l5m,#l6m,#l7m,#l8m{width:100%;padding:1em;border-radius: 1em;margin:5px 0;}\n";
  ptr += "#lp{display:flex;flex-wrap:wrap;justify-content:center;}\n";
  ptr += "input[type=\"radio\"]{width:25px;height:25px;}\ntable{margin:auto;border-collapse:collapse;}\ninput[type='text']{height:2rem;margin:5px 0;border-radius: 8px;padding: 4px;}\n";
  ptr += ".skipB{background-color:#00ffaa;color:white;padding:10px 20px;text-align:center;display:inline-block;margin:4px 2px;width:30%;}\n.skippedB{background-color:#ff0000;}\n";
  ptr += "#update{position:fixed;bottom:10px;left:50%;transform: translateX(-50%);box-shadow:0px 1px 7px 2px #988c8c;background:linear-gradient(to bottom, #ededed 5%, #c9ea83 100%);border-radius:6px;border:1px solid #dcdcdc;padding:6px 24px;font-size:1.0em;}";
  ptr += "</style>\n</head>\n<body>\n<h1 id='h1'>Auto Startz</h1>\n";
  ptr += "<button id=\"refetch\">Refetch</button>\n";
  if (rtcDetected) {
    ptr += "&#x23F2; " + String(RTC.getDay()) + "-" + String(RTC.getMonth()) + "-" + String(RTC.getYear()) + " " + String(RTC.getHours()) + ":" + String(RTC.getMinutes()) + ":" + String(RTC.getSeconds()) + " &nbsp Running: &nbsp" + String(minutes);
  } else {
    ptr += "&#x23F2; MOCK: " + String(mock_Hr) + ":" + String(mock_Min) + " &nbsp Running: &nbsp" + String(minutes);
  }
  ptr += "<button id=\"timeset\">update TIME</button>\n";
  ptr += "<div class='_Panel'>";
  for(uint8_t i=0; i<8 ; ++i){
    ptr += "<input type='checkbox' ";
    if(eepromVar2.displayingPanels[i] == '1') {
      ptr += "checked";
    }
    ptr += " id='panel" + String(i+1) + "' name='ledPANELS' /><label for='panel" + String(i+1) + "'>" + String(i+1) + "</label>\n";
  }
  ptr += dE + "\n";
  ptr += "<div id=\"lp" + tag_ + "\n";
  for(uint8_t i=0, ledCoupleCount=0; i<8 ; ++i, ledCoupleCount=i/2){
    if (eepromVar2.displayingPanels[i] == '1') {
      ptr += buildLedHtml(i, eepromVar1.ledCouple[ledCoupleCount], eepromVar1.isLedRtcInterval[i]);
    }
  }
  ptr += dE + "\n" + BI + UP + "\">" + UP + " to device" + BE + "\n";  //parent div
  ptr += "<div>RTC: HH.MM-HH.MM; Interval ON Seconds- OFF Seconds"+dE+"\n";         //parent div
  ptr += renderTable(true);
  ptr += "<script>\n";
  ptr += "var GI=e=>document.getElementById(e),QS=(e,t)=>GI(e).querySelector(t),QSA=e=>document.querySelectorAll(e),ids=Array.from(document.getElementsByName('ledPANELS')).filter(e=>e.checked).map(e=>e.id.replace('panel','l')),cc='';function stc(){const e=(new Date).toString();let t=0;e.split('').forEach(e=>{t=e.charCodeAt(0)+((t<<5)-t)});let n='#';for(let e=0;e<3;e++){n+=(t>>8*e&255).toString(16).padStart(2,'0')}return n}ids.forEach((e,t,n)=>{console.log(e,t,n),QS(e,'#led'+e[1]+'_off').checked?('OFF'==GI(e+'m').textContent&&(GI(e+'r').disabled=!0),QS(e,'#l'+e[1]+'cp')&&(QS(e,'#l'+e[1]+'cp').style.display='none',QS(e,'#l'+e[1]+'cp').nextSibling.textContent=''),QS(e,'#l'+e[1]+'rtccp')&&(QS(e,'#l'+e[1]+'rtccp').style.display='none',QS(e,'#l'+e[1]+'rtccp').nextSibling.textContent='')):GI(e+'m').style.display='none',QS(e,'#led'+e[1]+'_rtc').checked&&QS(e,'#l'+e[1]+'rtccp')&&(QS(e,'#l'+e[1]+'rtccp').style.display='none',QS(e,'#l'+e[1]+'rtccp').nextSibling.textContent=''),QS(e,'#led'+e[1]+'_interval').checked,GI(e+'r').addEventListener('input',n=>{QS(e,'#led'+e[1]+'_off').checked&&n.target.value&&'ON'==GI(e+'m').textContent?GI(e+'a').href='/led'+(t+1)+'on?time='+n.target.value:QS(e,'#led'+e[1]+'_off').checked&&!n.target.value&&'ON'==GI(e).textContent&&(GI(e+'a').href='/led'+(t+1)+'on')}),t%2==0&&GI(e+'cp').addEventListener('change',e=>{e.target.checked?(GI(ids[t+1]).style.border='5px solid red',GI(ids[t+1]).style.contentVisibility='hidden'):(GI(ids[t+1]).style.border=null,GI(ids[t+1]).style.contentVisibility='visible')}),'ON'==GI(e+'m').textContent?GI(e+'m').style.backgroundColor='green':GI(e+'m').style.backgroundColor='red'}),GI('h1').addEventListener('click',()=>{cc=stc(),ids.forEach(e=>{GI(e).style.backgroundColor=cc})});const RB=QSA('input[type=\"radio\"]'),homeBtn=GI('refetch'),updateBtn=GI('update'),TS=GI('timeset');RB.forEach(e=>{e.addEventListener('change',e=>{const t=e.target.value,n=t.slice(2);if(!['r','o','i','ip','c'].includes(n))return;const c=GI('l'+t[1]+'m'),l=GI('l'+t[1]+'r'),o=t.slice(0,2),i=QS(o,'#l'+t[1]+'cp'),r=QS(o,'#l'+t[1]+'rtccp');if(c.style.display='o'===n?'block':'none','o'!==n&&(l.style.display='block'),i){const e='r'===n||'i'===n;i.style.display=e?'block':'none',i.nextSibling.textContent=e?'Coupled':''}if(r){const e='i'===n;r.style.display=e?'block':'none',r.nextSibling.textContent=e?'use RTC range':''}l.value={r:'20.05-21.00',o:'5',i:'100-3600',ip:'#1-',c:'++'}[n]})}),updateBtn.addEventListener('click',()=>{var e={coupled:'',cc:cc,rtcinterval:'',panels:''};QSA('#lp input[type=\"text\"]').forEach(function(n,c){t=n.parentNode.querySelector('div.rdiv input#led'+(c+1)+'_input'),n.value||t.checked?e[n.name]=n.value:QS(n.parentNode.id,'#led'+(c+1)+'_off').checked?e[n.name]='-':e[n.name]=''}),QSA('input[name=\"ledcouple\"]').forEach((t,n,c)=>{t.checked&&(e['l'+(2+2*n)+'r']=t.parentNode.querySelector('#l'+(1+2*n)+'r').value),e.coupled+=t.checked?1:0}),QSA('input[name=\"ledRTCI\"]').forEach(t=>{e.rtcinterval+=t.checked?1:0}),QSA('input[name=\"ledPANELS\"]').forEach(t=>{e.panels+=t.checked?1:0});var t=confirm(JSON.stringify(e));const n=new URLSearchParams(e).toString();t&&fetch(`/eepromSet?${n}`)}),homeBtn.addEventListener('click',()=>window.location.href='/'),TS.addEventListener('click',()=>{var e=(new Date).toString().split(' '),t=e[0]+' '+e[1]+' '+e[2]+' '+e[4]+' '+e[3],n=confirm(JSON.stringify(t));const c=new URLSearchParams({date:t}).toString();n&&fetch(`/updateTime?${c}`)}),setTimeout(()=>{'/'!=window.location.pathname&&'localhost'!=window.location.hostname&&(window.location.href='/')},2e3);";
  ptr += "\n</script>\n</body>\n</html>\n";
  return ptr;
}

void handle_OnConnect() {
  if (server.hasArg("ui") || eepromVar2.showUiDefault == true) {
    server.send(200, "text/html", SendHTML());
  }else{
    server.send(200, "text/html", renderTable());
  }
}

void hanleConfigUpdate_coupled(String coupledValArg) {
  if (coupledValArg.length() == 4) {
    int rcoupledValArg = strtol(coupledValArg.c_str(), NULL, 2);
    for(uint8_t i=0; i<4 ; i++){
      eepromVar1.ledCouple[i] = (rcoupledValArg & (1 << (3 - i))) != 0;
    }
  } else {
    for(uint8_t i=0; i<4 ; i++){
      eepromVar1.ledCouple[i] = false;
    }
  }
}

void hanleConfigUpdate_rtc_interval(String useRtcIntervalFlagArg) {
  int rtcIntervalArg = strtol(useRtcIntervalFlagArg.c_str(), NULL, 2);
  //check RTC interval
  for (uint8_t i = 0; i < 8; i++) {
    // Check if the (7 - i)th bit is set
    eepromVar1.isLedRtcInterval[i] = (rtcIntervalArg & (1 << (7 - i))) != 0;
  }
}

void hanleConfigUpdate_panel(String displayingPanelArg) {
  bool isBinary = true;
  for (uint8_t i = 0; i < displayingPanelArg.length(); i++) {
    char c = displayingPanelArg.charAt(i);
    if (c != '0' && c != '1') {
      isBinary = false;
      break;
    }
  }
  if (!isBinary) {
    Serial.println("Invalid Panel configuration: " + displayingPanelArg );
    return;
  }

  if (displayingPanelArg.length() != 8) {
    // If the input string is not 8 characters long, set rest of the panels to hidden
    uint8_t userConfiged = displayingPanelArg.length()-1;
    for (uint8_t i = userConfiged; i < 8; i++) {
      eepromVar1.displayingPanels[i] = '0';
    }
    return;
  }
  Serial.println("Updating Panel configuration: " + displayingPanelArg + "length: " + String(displayingPanelArg.length()));
  int displayingPanelArgInt = strtol(displayingPanelArg.c_str(), NULL, 2);
  // for an example 11000000, the first two panels will be displayed and the rest will be hidden
  for (uint8_t i = 0; i < 8; i++) {
    eepromVar1.displayingPanels[i] = (displayingPanelArgInt & (1 << (7 - i))) != 0 ? '1' : '0';
  }
}

void hanleConfigUpdate_rtc_timing(uint8_t i,String rtcIntervalTimingValArg) {
  uint8_t lXhyphenIndx = rtcIntervalTimingValArg.indexOf("-", 1);
  String lXtimevalue1 = rtcIntervalTimingValArg.substring(0, lXhyphenIndx);
  String lXtimevalue2 = rtcIntervalTimingValArg.substring(lXhyphenIndx + 1);
  if (rtcIntervalTimingValArg.equals("-")) {
    eepromVar1.isLedTimer[i] = false;
    eepromVar1.isLedInterval[i] = false;
  } else if (lXtimevalue1.length() == 5 && lXtimevalue1.charAt(2) == '.' && lXtimevalue2.length() == 5 && lXtimevalue2.charAt(2) == '.' ){
    eepromVar1.isLedTimer[i] = true;
    eepromVar1.isLedInterval[i] = false;
    eepromVar1.ledonHr[i] = lXtimevalue1.substring(0, 2).toInt() > 23 ? 23 : lXtimevalue1.substring(0, 2).toInt();
    eepromVar1.ledonMin[i] = lXtimevalue1.substring(3).toInt() > 59 ? 59 : lXtimevalue1.substring(3).toInt();
    eepromVar1.ledoffHr[i] = lXtimevalue2.substring(0, 2).toInt() > 23 ? 23 : lXtimevalue2.substring(0, 2).toInt();
    eepromVar1.ledoffMin[i] = lXtimevalue2.substring(3).toInt() > 59 ? 59 : lXtimevalue2.substring(3).toInt();
  } else if (lXtimevalue1.length() && lXtimevalue2.length()) {
    eepromVar1.isLedTimer[i] = false;
    eepromVar1.isLedInterval[i] = true;
    eepromVar1.ledInterval_onMin[i] = lXtimevalue1.toInt();
    eepromVar1.ledInterval_offMin[i] = lXtimevalue2.toInt();
  }
  eepromVar1.gpio_input_map[i].pin = -1;
  eepromVar1.pulseCounterInputPinMap[i] = false;
}

void hanleConfigUpdate_input_gpio_pin_mapping(uint8_t i,String gpio_input_mapping_val_arg){ //example gpio_input_mapping_val_arg value received = #1-9
  eepromVar1.isLedTimer[i] = false;
  eepromVar1.isLedInterval[i] = false;
  
  gpio_input_mapping_val_arg = gpio_input_mapping_val_arg.substring(1); //removed prefix #
  uint8_t lXhyphenIndx = gpio_input_mapping_val_arg.indexOf("-", 1); //splited by -
  uint8_t gpio_pullup_or_pulldown = -1; //should be 1 or 0 
  uint8_t trigger_gpio_number = -1; //should be 0 to 9

  //check if both values are having expected values
  if(gpio_input_mapping_val_arg.substring(0, lXhyphenIndx) == "1" || gpio_input_mapping_val_arg.substring(0, lXhyphenIndx) == "0"){
    gpio_pullup_or_pulldown = gpio_input_mapping_val_arg.substring(0, lXhyphenIndx).toInt();
  }else{
    return;
  }

  if(gpio_input_mapping_val_arg.substring(lXhyphenIndx+1) > "0" && gpio_input_mapping_val_arg.substring(lXhyphenIndx+1) <= "9"){
    trigger_gpio_number = gpio_input_mapping_val_arg.substring(lXhyphenIndx+1).toInt();
  }else{
    return;
  }

  if(gpio_pullup_or_pulldown != -1 && trigger_gpio_number != -1){
    eepromVar1.gpio_input_map[i].is_pullup_pulldown = (bool)!!gpio_pullup_or_pulldown;
    eepromVar1.gpio_input_map[i].pin = trigger_gpio_number;
    // eepromVar1.gpio_input_map[i].trigger_pin = trigger_gpio_number;
  }else{
    eepromVar1.gpio_input_map[i].pin = -1;
  }
}

void hanleConfigUpdate_input_counter(u_int8_t i) {
  eepromVar1.pulseCounterInputPinMap[i] = true;
}

void hanleConfigUpdate_gpio_pin_mapping(
  uint8_t pinArray[8]
){
    uint8_t ledAssign[8]={-1,-1,-1,-1,-1,-1,-1,-1};
    bool ifAllArgsValidVal=true;
    for (uint8_t i = 0; i < 8; i++) {
      Serial.println("check1");
      uint8_t argVal = pinArray[i];
        if(argVal < 0 || argVal> 9){
          Serial.println("check2");
          break;
          Serial.println("check3");
          ifAllArgsValidVal=false;
        }
        if( argVal == 1 || argVal == 2 || argVal >9){
          Serial.println("check4");
          ifAllArgsValidVal=false;
        }
        ledAssign[i]=argVal;
        Serial.println("check5");
        for (uint8_t j = 0; j < i; j++){
          if(ledAssign[j]==argVal || ledAssign[i]== -1){
            Serial.println("check6");
            ifAllArgsValidVal=false;
            break;
          }
        }
        Serial.println("check7");
        if(!ifAllArgsValidVal){
          Serial.println("check8");
          break;
        }
    }
    if(ifAllArgsValidVal){
      Serial.println("going for reassign.....");
      char ledSeqVal[9];
      for (uint8_t i = 0; i < 8; i++) {
        ledSeqVal[i] = '0' + pinArray[i];
      }
      ledSeqVal[8] = '\0';
      reAssignLedOutputSequence(ledSeqVal);
      strcpy(eepromVar1.ledSquence, ledSeqVal);
    }
}

void hanleConfigUpdate() {
  received_cnf_res.clear();
  for (uint8_t i = 0; i < server.args(); i++) {
    received_cnf_res += server.argName(i); 
    received_cnf_res += "="; 
    received_cnf_res += server.arg(i);
    received_cnf_res += "\n"; 
  }

  // curl "http://192.168.1.1/eepromSet?coupled=1010
  if(server.hasArg("coupled")){
    hanleConfigUpdate_coupled(server.arg("coupled"));
  }

  // curl "http://192.168.1.1/eepromSet?coupled=10101010
  if(server.hasArg("rtcinterval")){
    hanleConfigUpdate_rtc_interval(server.arg("rtcinterval"));
  }

  // curl "http://192.168.1.1/eepromSet?panel=10101010
  if(server.hasArg("panels")){
    hanleConfigUpdate_panel(server.arg("panels"));
  }

  // curl "http://192.168.1.1/eepromSet?coupled=1010&l1r=300-3600&l2r=12.59-15.00&l3r=-&l4r=-&l5r=-&l6r=-&l7r=-&l8r=-
  // curl "http://192.168.1.1/eepromSet?l1r=#1-2 //
  // curl "http://192.168.1.1/eepromSet?l1r=++1 -> means input 1 will be treated as a counter to read pulses //
  String argsList[8]={"l1r", "l2r", "l3r", "l4r", "l5r", "l6r", "l7r", "l8r"};
  for(uint8_t i=0, ledCoupleCount=0; i<8; i=i+2, ledCoupleCount++){ //handling i & i+1 at a time
    bool isInputFlag =false;
    if (server.hasArg(argsList[i]) && server.arg(argsList[i]).charAt(0) == '#' ){
      hanleConfigUpdate_input_gpio_pin_mapping(i,server.arg(argsList[i])); 
      isInputFlag = true;
    } 
    if (server.hasArg(argsList[i+1]) && server.arg(argsList[i+1]).charAt(0) == '#' ){
      hanleConfigUpdate_input_gpio_pin_mapping(i+1,server.arg(argsList[i+1])); 
      isInputFlag=true;
    } 
    //check counter config
    if (server.hasArg(argsList[i]) && server.arg(argsList[i]).charAt(0) == '+' && server.arg(argsList[i]).charAt(1) == '+' ){
      hanleConfigUpdate_input_counter(i); 
      isInputFlag=true;
    } 
    if (server.hasArg(argsList[i+1]) && server.arg(argsList[i+1]).charAt(0) == '+' && server.arg(argsList[i+1]).charAt(1) == '+' ){
      hanleConfigUpdate_input_counter(i+1); 
      isInputFlag=true;
    } 
    if (isInputFlag){continue;}
    if (server.hasArg(argsList[i])) hanleConfigUpdate_rtc_timing(i,server.arg(argsList[i])); 
    if (server.hasArg(argsList[i+1]) && server.arg("coupled").charAt(ledCoupleCount) == '1'){
      hanleConfigUpdate_rtc_timing(i+1,server.arg(argsList[i])); 
    } else if(server.hasArg(argsList[i+1])) {
      hanleConfigUpdate_rtc_timing(i+1,server.arg(argsList[i+1]));
    } 
  }

  // curl "http://192.168.1.1/eepromSet?ui=1"
  eepromVar1.showUiDefault = server.hasArg("ui");

  // curl "http://192.168.1.1/eepromSet?cc=F0F0F0
  if (server.arg("cc").length() == 7) {
    strcpy(eepromVar1.colorCode, server.arg("cc").c_str());
  }
  if(server.hasArg("currentSensorIO") &&
    server.arg("currentSensorIO").length() == 2 && 
    server.arg("currentSensorIO").toInt() > 11 && 
    server.arg("currentSensorIO").charAt(0) != server.arg("currentSensorIO").charAt(1)
  ){
    eepromVar1.currentSensorrelatedIOs = server.arg("currentSensorIO").toInt(); //op 3 - makes contactor on; 4 - makes contactor OFF
  }
  if(server.hasArg("currentSensorMax") && server.arg("currentSensorMax").length()){
    eepromVar1.currentSensorThreshold = server.arg("currentSensorMax").toFloat(); 
  }

  //curl "http://192.168.1.1/eepromSet?l1=4&l2=0&l3=3&l4=5&l5=6&l6=7&l7=8&l8=9" -> this sets D0 to D9 based on the choice of outputs
  String argsList2[8]={"l1", "l2", "l3", "l4", "l5", "l6", "l7", "l8"};
  bool checkAllArgsPresent = true;
  for (uint8_t i = 0; i < 8; i++) {
    if (!(server.hasArg(argsList2[i]))) {
      checkAllArgsPresent = false;
      break;
    }
  }
  if(checkAllArgsPresent){
    uint8_t pinArray[8];
    for (uint8_t i = 0; i < 8; i++) {
      pinArray[i] = server.arg(argsList2[i]).toInt();
    }
    hanleConfigUpdate_gpio_pin_mapping(pinArray);
  }
  
  allGpioOff();

  eepromVar1.overCurrentDetected=false; //this makes emergency stop contactor reset when over current was triggered previously
  Serial.println("Writing to EEPROM..");
  // set the EEPROM data ready for writing
  EEPROM.put(0, eepromVar1);
  // write the data to EEPROM
  boolean ok = EEPROM.commit();
  Serial.println((ok) ? "Commit OK" : "Commit failed");
  eepromVar2 = eepromVar1;

  server.send(200, "text/html", renderTable());
}

void reAssignLedOutputSequence(String sequence){
  Serial.println(String("going for reassign....."+sequence));

  LED1pin = pinPool[(int)sequence.charAt(0)-'0']; 
  LED2pin = pinPool[(int)sequence.charAt(1)-'0']; 
  LED3pin = pinPool[(int)sequence.charAt(2)-'0']; 
  LED4pin = pinPool[(int)sequence.charAt(3)-'0']; 
  LED5pin = pinPool[(int)sequence.charAt(4)-'0']; 
  LED6pin = pinPool[(int)sequence.charAt(5)-'0']; 
  LED7pin = pinPool[(int)sequence.charAt(6)-'0']; 
  LED8pin = pinPool[(int)sequence.charAt(7)-'0']; 
  pinMode(LED1pin, OUTPUT);
  pinMode(LED2pin, OUTPUT);
  pinMode(LED3pin, OUTPUT);
  pinMode(LED4pin, OUTPUT);
  pinMode(LED5pin, OUTPUT);
  pinMode(LED6pin, OUTPUT);
  pinMode(LED7pin, OUTPUT);
  pinMode(LED8pin, OUTPUT);
}
void clearIntervalOrTimer(uint8_t ledNo) {  //pass index, start from 0
  eepromVar2.isLedTimer[ledNo] = false;
  eepromVar2.isLedInterval[ledNo] = false;
}

void ledOffImplementation(uint8_t ledNo) {  //pass index, start from 0
  endStopTimer[ledNo] = 0;
  LEDstatus[ledNo] = false;
  Serial.println("LED" + String(ledNo) + " Maually OFF");
  // fetching original seeting FROM EEPROM variable, user made off, so next time automatic system should take up
  eepromVar2.ledInterval_onMin[ledNo] = eepromVar1.ledInterval_onMin[ledNo];
  eepromVar2.ledInterval_offMin[ledNo] = eepromVar1.ledInterval_offMin[ledNo];
  eepromVar2.isLedTimer[ledNo] = eepromVar1.isLedTimer[ledNo];
  eepromVar2.isLedInterval[ledNo] = eepromVar1.isLedInterval[ledNo];
  server.send(200, "text/html", renderTable());
}

void handle_ledon(uint8_t ledNo) {  //pass index, start from 0
  clearIntervalOrTimer(ledNo); //Passing Index for leds //TODO: refine
  if (server.hasArg("time")) {
    endStopTimer[ledNo] = server.arg("time").toInt();
  }
  LEDstatus[ledNo] = true;
  Serial.println("LED"+String(ledNo+1)+" Maually ON");
  server.send(200, "text/html", renderTable());
}
void handle_ledoff(uint8_t ledNo) { //pass index, start from 0
  ledOffImplementation(ledNo);  //pass index
}


void handleSkipAuto() {
  if (server.hasArg("led")) {
    uint8_t ledNo = server.arg("led").toInt();
    isLedSkippedAuto[ledNo - 1] = true;//here ledno starts from 1
    if((ledNo - 1)%2==0 && eepromVar1.ledCouple[(ledNo - 1)/2]){ //led couple pair1 is skipped, then skip the 2nd also. if led pair2 is skipped, then skip pair1 also
      isLedSkippedAuto[ledNo] = true; //make the 2nd pair also skipped
    }else if((ledNo - 1)%2!=0 && eepromVar1.ledCouple[(ledNo - 1)/2]){
      isLedSkippedAuto[ledNo-2] = true; //make the 2nd pair also skipped
    }
  }
  server.send(200, "text/html", renderTable());
}
void handleUnskipAuto() {
  if (server.hasArg("led")) {
    uint8_t ledNo = server.arg("led").toInt();
    isLedSkippedAuto[ledNo - 1] = false;//here ledno starts from 1
    if((ledNo - 1)%2==0 && eepromVar1.ledCouple[(ledNo - 1)/2]){ //led couple pair1 is unskipped, then unskip the 2nd also. if led pair2 is unskipped, then unskip pair1 also
      isLedSkippedAuto[ledNo] = false; //make the 2nd pair also unskipped
    }else if((ledNo - 1)%2!=0 && eepromVar1.ledCouple[(ledNo - 1)/2]){
      isLedSkippedAuto[ledNo-2] = false; //make the 2nd pair also unskipped
    }
  }
  server.send(200, "text/html", renderTable());
}

void handleTimeUpdate() {
  if (server.hasArg("date")) {
    String date = String(server.arg("date"));
    // String time = String(server.arg("time"));
    // char* d;char* t;
    // date.toCharArray(d, date.length()+1); time.toCharArray(t, time.length()+1);
    // Serial.println(date+" "+time);
    // Serial.println(d);

    //"Sat Apr 26 13:36:18 2025"
    // Find the first colon (hours separator)
    int firstColonIndex = date.indexOf(':');
    // Extract hour by taking the two characters before the first colon
    String hour = date.substring(firstColonIndex - 2, firstColonIndex);
    // Find the second colon (minutes separator)
    String mins = date.substring(firstColonIndex + 1, firstColonIndex + 3);
    mock_Hr = hour.toInt();
    mock_Min = mins.toInt();

    Serial.println(date);
    RTC.setDateTime(date);
  }
  server.send(200, "text/html", renderTable());
}

void registerServerWithCallbackFunctions() {
  server.on("/", handle_OnConnect);
  server.on("/eepromSet", hanleConfigUpdate);
  server.on("/led1on", [](){ handle_ledon(0); });
  server.on("/led1off", [](){ handle_ledoff(0); });
  server.on("/led2on", [](){ handle_ledon(1); });
  server.on("/led2off", [](){ handle_ledoff(1); });
  server.on("/led3on", [](){ handle_ledon(2); });
  server.on("/led3off", [](){ handle_ledoff(2); });
  server.on("/led4on", [](){ handle_ledon(3); });
  server.on("/led4off", [](){ handle_ledoff(3); });
  server.on("/led5on", [](){ handle_ledon(4); });
  server.on("/led5off", [](){ handle_ledoff(4); });
  server.on("/led6on", [](){ handle_ledon(5); });
  server.on("/led6off", [](){ handle_ledoff(5); });
  server.on("/led7on", [](){ handle_ledon(6); });
  server.on("/led7off", [](){ handle_ledoff(6); });
  server.on("/led8on", [](){ handle_ledon(7); });
  server.on("/led8off", [](){ handle_ledoff(7); });
  server.on("/serial", [](){ serialPrintEnable=!serialPrintEnable; server.send(200, "text/html", renderTable()); });
  server.on("/updateTime", handleTimeUpdate);
  server.on("/ledskipauto", handleSkipAuto);
  server.on("/ledunskipauto", handleUnskipAuto);
  // server.onNotFound(handle_NotFound);
}

void checkWhichLedSurpassRTC(int RTC_Hours, int RTC_Minutes) {
  //check led8 config with rtc
  for(uint8_t i=0; i<8 ; i++){
    if (eepromVar2.isLedTimer[i] && (eepromVar2.ledonHr[i] < eepromVar2.ledoffHr[i])) {  //check if on hr is less than offhour; means day time
      if (((eepromVar2.ledonHr[i] == RTC_Hours && RTC_Minutes >= eepromVar2.ledonMin[i]) || (RTC_Hours > eepromVar2.ledonHr[i])) && ((RTC_Hours < eepromVar2.ledoffHr[i]) || (eepromVar2.ledoffHr[i] == RTC_Hours && RTC_Minutes < eepromVar2.ledoffMin[i]))) {
        //turn on the led 8
        if (!isLedSkippedAuto[i]) LEDstatus[i] = true;
      } else {
        //turn off the led 8
        if (!isLedSkippedAuto[i]) LEDstatus[i] = false;
      }
    } else if (eepromVar2.isLedTimer[i] && (eepromVar2.ledonHr[i] > eepromVar2.ledoffHr[i])) {  //check if on hr is greater than offhour; means night time
      if (/*  */ (eepromVar2.ledonHr[i] == RTC_Hours && RTC_Minutes >= eepromVar2.ledonMin[i]) || (RTC_Hours > eepromVar2.ledonHr[i]) || ((eepromVar2.ledoffHr[i] == RTC_Hours && RTC_Minutes < eepromVar2.ledoffMin[i]) || (RTC_Hours < eepromVar2.ledoffHr[i])) /*  */) {
        //turn on the led 8
        if (!isLedSkippedAuto[i]) LEDstatus[i] = true;
      } else {
        //turn off the led 8
        if (!isLedSkippedAuto[i]) LEDstatus[i] = false;
      }
    } else if (eepromVar2.isLedTimer[i] && (eepromVar2.ledonHr[i] == eepromVar2.ledoffHr[i])) {  //check if on hr is equal offhour; means same hour
      if (RTC_Minutes >= eepromVar2.ledonMin[i] && RTC_Minutes < eepromVar2.ledoffMin[i] && RTC_Hours == eepromVar2.ledonHr[i]) {
        //turn on the led 8
        if (!isLedSkippedAuto[i]) LEDstatus[i] = true;
      } else {
        //turn off the led 8
        if (!isLedSkippedAuto[i]) LEDstatus[i] = false;
      }
    }
  }
}

bool checkIntervalRtcRange(int RTC_Hours, int RTC_Minutes, int setRtcStartHr, int setRtcStartMin, int setRtcEndHr, int setRtcEndMin) {
  if (setRtcStartHr < setRtcEndHr) {  //check if on hr is less than offhour; means day time
    if (((setRtcStartHr == RTC_Hours && RTC_Minutes >= setRtcStartMin) || (RTC_Hours > setRtcStartHr)) && ((RTC_Hours < setRtcEndHr) || (setRtcEndHr == RTC_Hours && RTC_Minutes < setRtcEndMin))) {
      return true;
    } else {
      return false;
    }
  } else if (setRtcStartHr > setRtcEndHr) {  //check if on hr is greater than offhour; means night time
    if (/*  */ (setRtcStartHr == RTC_Hours && RTC_Minutes >= setRtcStartMin) || (RTC_Hours > setRtcStartHr) || ((setRtcEndHr == RTC_Hours && RTC_Minutes < setRtcEndMin) || (RTC_Hours < setRtcEndHr)) /*  */) {
      return true;
    } else {
      return false;
    }
  } else if (setRtcStartHr == setRtcEndHr) {  //check if on hr is equal offhour; means same hour
    if (RTC_Minutes >= setRtcStartMin && RTC_Minutes < setRtcEndMin && RTC_Hours == setRtcStartHr) {
      return true;
    } else {
      return false;
    }
  }
  return false;
}

void intervalNonCoupleOprtaionImpl(uint8_t ledNo, int RTC_Hours, int RTC_Minutes) {
  if (eepromVar1.isLedInterval[ledNo]) {
    if(eepromVar1.isLedRtcInterval[ledNo]){
      bool isRTCRangeOk = checkIntervalRtcRange(RTC_Hours, RTC_Minutes, eepromVar1.ledonHr[ledNo], eepromVar1.ledonMin[ledNo], eepromVar1.ledoffHr[ledNo], eepromVar1.ledoffMin[ledNo]);
      if (!isRTCRangeOk){
        LEDstatus[ledNo] = false;
        return;
      }
    }
    if (LEDstatus[ledNo]) eepromVar2.ledInterval_onMin[ledNo]=eepromVar2.ledInterval_onMin[ledNo]-3;  // we are using 3 seconds to watch led status As
    if (!LEDstatus[ledNo]) eepromVar2.ledInterval_offMin[ledNo]=eepromVar2.ledInterval_offMin[ledNo]-3;
    if (LEDstatus[ledNo] && eepromVar2.ledInterval_onMin[ledNo] <= 0) {
      //turn off the led 1
      LEDstatus[ledNo] = false;
      eepromVar2.ledInterval_onMin[ledNo] = eepromVar1.ledInterval_onMin[ledNo]; //Reload the decremented counter
    }
    if (!LEDstatus[ledNo] && eepromVar2.ledInterval_offMin[ledNo] <= 0) {
      //turn on the led 1
      LEDstatus[ledNo] = true;
      eepromVar2.ledInterval_offMin[ledNo] = eepromVar1.ledInterval_offMin[ledNo];
    }
  }
}

void checkLedIntevalSequence(int RTC_Hours, int RTC_Minutes) {
  /* 
  iteration: 1st_led 2nd_led - formula_for_1st_led formula_for_2st_led
  0: 0 1 - 0X2 0X2+1
  1: 2 3 - 1X2 1X2+1
  2: 4 5 - 2X2 2X2+1
  3: 6 7 - 3X2 3X2+1
  */
  for(uint8_t i=0; i<4 ; i++){
    uint8_t firstLedIndex = i*2;
    uint8_t secondLedIndex = i*2+1;
    if (eepromVar1.isLedInterval[firstLedIndex] && eepromVar1.ledCouple[i]) {//remember in couple 1st led configuration is read by the second led.
      if(eepromVar1.isLedRtcInterval[firstLedIndex]){ //if its supposed to follow RTC time to blink logic, then check
        bool isRTCRangeOk = checkIntervalRtcRange(RTC_Hours, RTC_Minutes, eepromVar1.ledonHr[firstLedIndex], eepromVar1.ledonMin[firstLedIndex], eepromVar1.ledoffHr[firstLedIndex], eepromVar1.ledoffMin[firstLedIndex]);
        if (!isRTCRangeOk){
          LEDstatus[firstLedIndex] = false;
          LEDstatus[secondLedIndex] = false;
          continue;
        }
      }
      if(eepromVar2.ledInterval_onMin[firstLedIndex] == eepromVar1.ledInterval_onMin[firstLedIndex] && eepromVar2.ledInterval_offMin[firstLedIndex] == eepromVar1.ledInterval_offMin[firstLedIndex]){ //first time when started, all timers have full value, no substracted....
        eepromVar2.ledInterval_onMin[firstLedIndex]--;
        eepromVar2.ledInterval_onMin[secondLedIndex]--;
        LEDstatus[firstLedIndex] = true;
        LEDstatus[secondLedIndex] = false;
        continue;
      }
      if (LEDstatus[firstLedIndex]) {
        eepromVar2.ledInterval_onMin[firstLedIndex]-=3;
        eepromVar2.ledInterval_onMin[secondLedIndex]-=3;
      }
      if (LEDstatus[secondLedIndex]) {
        eepromVar2.ledInterval_offMin[firstLedIndex]-=3;
        eepromVar2.ledInterval_offMin[secondLedIndex]-=3;
      }
      if (eepromVar2.ledInterval_onMin[firstLedIndex] <= 0) {
        LEDstatus[firstLedIndex] = false;
        LEDstatus[secondLedIndex] = true;
        eepromVar2.ledInterval_onMin[firstLedIndex] = eepromVar1.ledInterval_onMin[firstLedIndex];
        eepromVar2.ledInterval_onMin[secondLedIndex] = eepromVar1.ledInterval_onMin[secondLedIndex];
        eepromVar2.ledInterval_offMin[firstLedIndex]--;
        eepromVar2.ledInterval_offMin[secondLedIndex]--;
        continue;
      }
      if (eepromVar2.ledInterval_offMin[firstLedIndex] <= 0) {
        LEDstatus[firstLedIndex] = true;
        LEDstatus[secondLedIndex] = false;
        eepromVar2.ledInterval_offMin[firstLedIndex] = eepromVar1.ledInterval_offMin[firstLedIndex];
        eepromVar2.ledInterval_offMin[secondLedIndex] = eepromVar1.ledInterval_offMin[secondLedIndex];
        eepromVar2.ledInterval_onMin[firstLedIndex]--;
        eepromVar2.ledInterval_onMin[secondLedIndex]--;
      }

    }
  }
  for(uint8_t i=0,ledCoupleCount=0; i<8 ; ++i,ledCoupleCount=i/2){
      if (!(eepromVar1.ledCouple[ledCoupleCount]) && !isLedSkippedAuto[i]) {
        intervalNonCoupleOprtaionImpl(i, RTC_Hours, RTC_Minutes);
      }
  }
}

void check_pin_input_level(){
  for (uint8_t i = 0; i < 8; i++) {
    uint8_t pin_no = i ==0 ? LED1pin : (i ==1 ? LED2pin : (i ==2 ? LED3pin : (i ==3 ? LED4pin : (i ==4 ? LED5pin : (i ==5 ? LED6pin : (i ==6 ? LED7pin : LED8pin))) )
    ));
    
    if(eepromVar1.gpio_input_map[i].pin != -1){
      if(digitalRead(pin_no) == eepromVar1.gpio_input_map[i].is_pullup_pulldown){
        LEDstatus[eepromVar1.gpio_input_map[i].pin] = true;
      }else {
        LEDstatus[eepromVar1.gpio_input_map[i].pin] = false;
      }
    }
  }
}

void checkEndStopTimer() {
  for (uint8_t i = 0; i < 8; i++) {
    if (endStopTimer[i] > 1) {
      endStopTimer[i]--;
    } else if (endStopTimer[i] == 1) {
      endStopTimer[i] = 0;
      ledOffImplementation(i);  //pass index
      LEDstatus[i] = false;
    }
  }
}


void displayOLEDscreen(String *content) {
  static int scrollIndex[8] = {0};

  display.clearDisplay();

  for (uint8_t i = 0; i < 8; i++) {
    int start = scrollIndex[i];
    int end   = start + 21;
    if (end > content[i].length()) end = content[i].length();

    String visible = content[i].substring(start, end);
    display.setCursor(0, i*8);
    display.print(visible.c_str());

    scrollIndex[i]+=5;
    if (scrollIndex[i] > content[i].length() - 21) {
      scrollIndex[i] = 0; // loop back
    }
  }

  display.display();
}

void IRAM_ATTR pulseCounter(void* arg) {
  uint8_t index = (uint8_t)(uintptr_t)arg;
  pulseCounterVal[index]++;
  // Serial.print("pulseCounterVal:");
  // Serial.print(index);
  // Serial.print(" = ");
  // Serial.println((unsigned long)pulseCounterVal[index]);
}

void setup() {
  Serial.begin(115200);
  RTC.begin();
  RTC.setHourMode(CLOCK_H24);
  // RTC.setHourMode(CLOCK_H12);
  // RTC.setEpoch(1725108207);
  //// sample input: date = "Dec 26 2009", time = "12:34:56"
  // RTC.setDateTime(__DATE__, __TIME__);

  if(RTC.isConnected() && RTC.isRunning()){
    rtcDetected=true;
  }else {
    rtcDetected=false;
  }

  
  ledOpFlag = 1;
  
  setInitEEPROM();
  EEPROM.begin(sizeof(MyEEPROMStruct));
  checkEEPROM_percentage();
 
  reAssignLedOutputSequence(String(eepromVar1.ledSquence));

  if(eepromVar1.gpio_input_map[0].pin != -1 || eepromVar1.pulseCounterInputPinMap[0]){
    if(eepromVar1.pulseCounterInputPinMap[0]){
      attachInterruptArg(digitalPinToInterrupt(LED1pin), pulseCounter, (void*)0, RISING);
    }
    pinMode(LED1pin, INPUT_PULLUP);
  }else {
    pinMode(LED1pin, OUTPUT);
  }
  if(eepromVar1.gpio_input_map[1].pin != -1 || eepromVar1.pulseCounterInputPinMap[1]){
    if(eepromVar1.pulseCounterInputPinMap[1]){
      attachInterruptArg(digitalPinToInterrupt(LED2pin), pulseCounter, (void*)1, RISING);
    }
    pinMode(LED2pin, INPUT_PULLUP);
  }else {
    pinMode(LED2pin, OUTPUT);
  }
  if(eepromVar1.gpio_input_map[2].pin != -1 || eepromVar1.pulseCounterInputPinMap[2]){
    if(eepromVar1.pulseCounterInputPinMap[2]){
      attachInterruptArg(digitalPinToInterrupt(LED3pin), pulseCounter, (void*)2, RISING);
    }
    pinMode(LED3pin, INPUT_PULLUP);
  }else {
    pinMode(LED3pin, OUTPUT);
  }
  if(eepromVar1.gpio_input_map[3].pin != -1 || eepromVar1.pulseCounterInputPinMap[3]){
    if(eepromVar1.pulseCounterInputPinMap[3]){
      attachInterruptArg(digitalPinToInterrupt(LED4pin), pulseCounter, (void*)3, RISING);
    }
    pinMode(LED4pin, INPUT_PULLUP);
  }else {
    pinMode(LED4pin, OUTPUT);
  }
  if(eepromVar1.gpio_input_map[4].pin != -1 || eepromVar1.pulseCounterInputPinMap[4]){
    if(eepromVar1.pulseCounterInputPinMap[4]){
      attachInterruptArg(digitalPinToInterrupt(LED5pin), pulseCounter, (void*)4, RISING);
    }
    pinMode(LED5pin, INPUT_PULLUP);
  }else {
    pinMode(LED5pin, OUTPUT);
  }
  if(eepromVar1.gpio_input_map[5].pin != -1 || eepromVar1.pulseCounterInputPinMap[5]){
    if(eepromVar1.pulseCounterInputPinMap[5]){
      attachInterruptArg(digitalPinToInterrupt(LED6pin), pulseCounter, (void*)5, RISING);
    }
    pinMode(LED6pin, INPUT_PULLUP);
  }else {
    pinMode(LED6pin, OUTPUT);
  }
  if(eepromVar1.gpio_input_map[6].pin != -1 || eepromVar1.pulseCounterInputPinMap[6]){
    if(eepromVar1.pulseCounterInputPinMap[6]){
      attachInterruptArg(digitalPinToInterrupt(LED7pin), pulseCounter, (void*)6, RISING);
    }
    pinMode(LED7pin, INPUT_PULLUP);
  }else {
    pinMode(LED7pin, OUTPUT);
  }
  if(eepromVar1.gpio_input_map[7].pin != -1 || eepromVar1.pulseCounterInputPinMap[7]){
    if(eepromVar1.pulseCounterInputPinMap[7]){
      attachInterruptArg(digitalPinToInterrupt(LED8pin), pulseCounter, (void*)7, RISING);
    }
    pinMode(LED8pin, INPUT_PULLUP);
  }else {
    pinMode(LED8pin, OUTPUT);
  }
  
  allGpioOff();

  WiFi.softAP(ssid, password);
  WiFi.softAPConfig(local_ip, gateway, subnet);
  delay(100);

  registerServerWithCallbackFunctions();

  // Convert string to integer (hex) 
  int lcdAddr = (int) strtol(eepromVar1.lcdADDR, NULL, 16);
  if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println(F("SSD1306 allocation failed"));
  }else {
    Serial.print("LCD started at address = ");
    Serial.println(lcdAddr);
  }

  if (!ads.begin()) {
    Serial.println("Failed to initialize ADS.");
    adsDetected=false;
  }else {
    adsDetected=true;
  }

  display.clearDisplay();
  display.setTextSize(1);               // Smallest text size
  display.setTextColor(SSD1306_WHITE);  // White text
  display.setCursor(0,0);
  // display.println(F("123456789111315171921"));
  // display.println(F("223456789111315171921"));
  // display.println(F("323456789111315171921"));
  // display.println(F("423456789111315171921"));
  // display.println(F("523456789111315171921"));
  // display.println(F("623456789111315171921"));
  // display.println(F("723456789111315171921"));
  // display.println(F("823456789111315171921"));
  // display.display();
  // display.startscrollleft(0x00, 0x0F);
  // display.startscrollleft(0, 7);  // scroll whole screen

  server.begin();
}

void loop() {
  server.handleClient();

  timeNow = millis() / 1000;         // the number of milliseconds that have passed since boot
  int seconds = timeNow - timeLast;  //the number of seconds that have passed since the last time 60 seconds was reached.

  if (seconds >= 60) {
    // if(rtcDetected){
    //   Serial.print("Epoch=");
    //   Serial.print(RTC.getEpoch());
    //   Serial.print(" RTC=");
    //   Serial.print(String(RTC.getDay()));
    //   Serial.print("-");
    //   Serial.print(String(RTC.getMonth()));
    //   Serial.print("-");
    //   Serial.print(String(RTC.getYear()));
    //   Serial.print("  ");
    //   Serial.print(String(RTC.getHours()));
    //   Serial.print(":");
    //   Serial.print(String(RTC.getMinutes()));
    //   Serial.print(":");
    //   Serial.print(String(RTC.getSeconds()));
    // }

    timeLast = timeNow;
    minutes = minutes + 1;
    seconds = 0;
    //let the sketch know that a new day has started for what concerns correction, if this line was not here the arduiono
    // would continue to correct for an entire hour that is 24 - startingHour.
    Serial.print(" Timer=");
    Serial.print(minutes);
    Serial.print(":=>");
    Serial.println(seconds);

    mock_Min = mock_Min + 1;
    if (mock_Min > 59) {
      mock_Hr = mock_Hr + 1;
      mock_Min = 0;
    }
    if (mock_Hr > 23) {
      mock_Hr = 0;
    }
    // checkWhichLedSurpassRTC(RTC.getHours(), RTC.getMonth());
    checkEndStopTimer();
  }
  uint8_t checkval = seconds % 3;
  if ((checkval == 0) && ledOpFlag) {
    if (rtcDetected) {
      checkLedIntevalSequence(RTC.getHours(), RTC.getMinutes());  //every seconds as interval we are using seconds
      checkWhichLedSurpassRTC(RTC.getHours(), RTC.getMinutes());
    } else {
      checkLedIntevalSequence(mock_Hr, mock_Min);  //every seconds as interval we are using seconds
      checkWhichLedSurpassRTC(mock_Hr, mock_Min);
    }
    check_pin_input_level();
    writeToGpio();
    ledOpFlag = 0;
    displayLedStatus();
  } else if (checkval != 0) {
    ledOpFlag = 1;
  }

  if(!serialPrintEnable){
    Serial.end();
  }else {
    Serial.begin(115200);
  }

  if(adsDetected && LEDstatus[eepromVar1.currentSensorrelatedIOs/10-1]){
    adc0 = ads.computeVolts(ads.readADC_SingleEnded(0));
    if(adc0 > eepromVar1.currentSensorThreshold){
      eepromVar1.overCurrentDetected = true;
    }
  }

  if(eepromVar1.overCurrentDetected){
    LEDstatus[eepromVar1.currentSensorrelatedIOs/10-1]=false;
    uint8_t emergencyStopLed= (eepromVar1.currentSensorrelatedIOs%10)-1; //got index of that LED
    eepromVar1.ledCouple[emergencyStopLed/2]=false;
    eepromVar1.isLedTimer[emergencyStopLed]=false;
    eepromVar1.isLedInterval[emergencyStopLed]=false;
    eepromVar1.isLedRtcInterval[emergencyStopLed]=false;
    LEDstatus[emergencyStopLed]=true;
    eepromVar2 = eepromVar1;
  }
}
