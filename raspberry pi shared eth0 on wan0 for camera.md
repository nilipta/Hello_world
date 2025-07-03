sudo systemctl status NetworkManager

Copilot said: Great! Since your Raspberry Pi is using
Great! Since your Raspberry Pi is using NetworkManager, you should configure Internet sharing using its tools (nmcli or nmtui) instead of dhcpcd. Here’s how you can share your Wi-Fi (wlan0) Internet connection with devices connected to Ethernet (eth0):

1. Set Ethernet (eth0) to "Shared" mode with NetworkManager
Using nmcli (command line)
bash
sudo nmcli con add type ethernet ifname eth0 con-name shared-eth0 ipv4.method shared
This command:

Creates a new connection named shared-eth0 for interface eth0
Sets IPv4 method to shared, which enables NAT and DHCP for devices on Ethernet
If you already have an Ethernet connection, you can modify it instead:

bash
sudo nmcli con modify "Wired connection 1" ipv4.method shared
sudo nmcli con up "Wired connection 1"
(Replace "Wired connection 1" with the actual name, if different. List with nmcli con.)

Using nmtui (text menu interface)
Run sudo nmtui
Go to Edit a connection
Select your Ethernet connection (e.g., eth0)
Set IPv4 CONFIGURATION to Shared to other computers
Save and quit
Restart the NetworkManager or reconnect the interface:
bash
sudo nmcli con down "<connection name>" && sudo nmcli con up "<connection name>"
2. Make sure Wi-Fi (wlan0) is connected to the Internet
Use nmcli device status to check that wlan0 is connected.

3. Connect your device to Raspberry Pi Ethernet
Plug another device (PC, etc.) into the Pi’s Ethernet. It should get an IP (usually 10.42.0.x), gateway, and Internet via the Pi’s Wi-Fi.

4. (Optional) Reboot or restart NetworkManager
bash
sudo systemctl restart NetworkManager
That’s it!
No need for iptables or dnsmasq — NetworkManager handles NAT, DHCP, and routing.

If you want a script for this, let me know!
