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
