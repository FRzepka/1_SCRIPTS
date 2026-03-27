# MG-Farm-4s-BMS

The MG-Farm 4s BMS is a BMS solution developed to monitor batterie packs with 3 to 4 cells in series for aging tests as well as to provide a platform to develop and validate state of the art state estimation algorithms utilizing battery testers and hardware available at EET.

## Getting Started

To get started working with the BMS you will need:

- 1x Assembled BMS Board
- 1x ST-Link (optional for programming and debugging)
- 1x Pack with 4s cell configuration
- 1x M8 cable lug for the positive battery terminal
- up to 3x NTC Sensors
- 1x USB-RS485 Converter
- 1x USB-UART Converter (optional)
- 1x 5V Supply
- 1x 3D-Printed Holder
- 4x M3 Screws
- 1x M8 Screw and Nut

The assembled PCB is first mounted on the 3D Printed holder by using the 4 M3 Screws. In the next step, the Battery can be secured by using some zip ties. The Cell connector coming from the battery pack is connected to the 7-Pin connector marked with Cells. Make sure that the orientation of the cell connector is correct and the highest potential is on the right side of the connector, marked with + on the PCB. The NTC Sensors are connected in a similar way with the three 2-pin connectors next to the cell connector marked with TMP1, TMP2. and TMP3.

The Potential of these connectors is directly tied to the negative potential of the battery pack! For Power and communication, the connectors on the lower right side of the PCB are used. The 2-Pin connector is used as the 5V Power supply for the STM32, isolators etc, while the 4-pin connectors provide a RS485 and a UART Interface. Note that the supply as well as the interfaces are galvanically isolated from the rest of the PCB as well as each other. As a result, the *VCC* and *GND* Pins for the serial interfaces *are required* and not optional!

To programm the board, the SWD interface located on the 6-Pin 2.54mm header on the bottom of the PCB can be found. The Pinout follows the standard ST-Link layout, with pin 1 indicating the VDD_Target Pin of the ST-Link. The Debugging interface is *not* isolated from the battery potential.

After programming and turning on the PCB for the first time, the display should show informations about the cell voltage, temperature sensors, current as well as the systick of the last measurement in milliseconds. 

If everything works up to this point, the battery tester can be connected with the Battery. For that, the positive current wire of the cell tester is connected with the Load+ terminal of the PCB via a cable lug and an M8 screw. The Sense+ wire of the battery tester has to be connected with the Bat+ terminal together with the positive terminal cable coming from the battery. The negativ terminal of the battery is directly connected with the negative sense wire and negatice current wire from the battery tester.

![](/readme_res/pcb_rendering.PNG "BMS PCB top view")

## Repo organization

This repository is structured in three main subfolders:
- The Hardware folder holds all necessary PCB schematic, layout and production files
- The Firmware folder contains the STM32 Source Code and supporting files
- The Documentation Folder contains additional documentation

## Toolchains

To work on this project it is required to install a toolchain to work with STMs .ioc format. While the VSCode Plugin in combination with CubeMX is getting continously better, I would recommend installing the STM32 Cube IDE for working with the Firmware. Furthermore to view and edit the schematic and layout, a KiCAD 7 or newer installation is required. The 3D models for the stand are modelled in Fusion 360 but are included as .step and .stl in this repo so any 3D CAD programm should do the job. 

The logging Software for a connected PC is written in Python. Dependencys can be installed via pip using the requirements.txt file. Note that the serial interface used for the main bus between host and BMS boards is based on RS485 so a RS485 to USB converter is required for connecting to the bus.

### Cube IDE setup

The STM32H755 used in this project is a rather complex dual core mcu. This leads to some additional configuration required to set up the project to debug properly. Thankfully ST documented the process in [AN5361](https://www.st.com/resource/en/application_note/dm00629855-getting-started-with-projects-based-on-dualcore-stm32h7-microcontrollers-in-stm32cubeide-stmicroelectronics.pdf). Make sure to also deactivate the inital breakpoint in the debug configurations of both MCUs to prevent faults when starting the debug session via the debug group.

## Hardware Overview

### TLE9012DQU

The TLE9012DQU is the main BMS IC for the BMS. Its task is to provide analog to digital conversion of cell voltages, temperatures and current, as well as implementing part of the necessary circuitry for passive cell balancing. 

Documentation for this IC is scattered accross the [datasheet](https://www.infineon.com/dgdl/Infineon-TLE9012DQU-DataSheet-v01_00-EN.pdf?fileId=8ac78c8c7e7124d1017f0c3d27c75737) and the [user manual](https://www.infineon.com/dgdl/Infineon-Infineon-TLE9012DQU_TLE9015DQU-UM-v01_00-EN-UserManual-v01_00-EN.pdf?fileId=8ac78c8c7e7124d1017f0c4f8750574b&da=t). The datasheet provides mainly informations about the electrical characteristics of the chip while the user manual contains the register map, a reference schematic, some additional information and a few example commands. If there are some informations missing in both the user manual and datasheet, it usually is a good idea to check the datasheet of the rather similar predecessor [TLE9012AQU](https://www.mouser.com/datasheet/2/196/Infineon_TLE9012AQU_DataSheet_v01_10_EN-1890780.pdf). 

The PCB Schematic mainly follows the reference design outlined in the user manual with the main difference of reducing the number of connected cells from 12 to just 4. Unused channels are connected with GND potential. In addition, the single wire UART interface is used instead of the ISO-UART which is used in the reference design. To allow for bidirectional current measurements, the TMP3 and TMP4 channel is used with the BAVM function of the TLE9012. The BAVM function allows to repurpose the 16bit ADC used for independintly measuring the sum voltage of all cells as a bipolar ADC on inputs TMP3 and TMP4.

### STM32

To ensure enough computation power to test modern state estimation techniques, a dual core STM32 microcontroller is used offering an ARM Cortex M4 and M7 core, with 1Mbyte of Flash per Core as well as 1Mbyte of SRAM.

The schematic of the microcontroller section is a bare bone implementation of the STM32H755 as described in the datasheet. The H755 offers the possibility to increase efficency by using some external components to form a switching mode regulator. Due to reasons of simplicity, this is not implemented in this design and the internal linear regulator is used to supply the core voltage.

In terms of communication interfaces, three UARTs of the STM32 are in use, UART Peripheral 1 is used as an isolated UART which can be connected to a UART-USB connector. For communication between a host and multiple BMS boards, UART3 is used in combination with an isolated RS485 transceiver which implements a custom serial protocol. The third UART is used in single wire mode to communicate with the TLE9012DQU (see the known quirks and errors section for further information). 

### Power Supply Scheme

While the TLE9012DQU supplies itself out of the battery, the STM32 would overload the internal regulator of the TLE so an independet power source is necessary. The supply is therefore ensured by using an isolated DC/DC 5V to 5V converter with the primary side supplied by the Host. The DC/DC itself is unregulated and has an ensured output voltage of at least 4.5V. Therefore two independent 3V3 regulators are used to supply digital and analog ICs with 3V3.

To monitor the current consumption of the STM32 and supporting ICs, it is possible to optionally utilize a shunt and a current sense amplifier connected to the ADC peripheral of the STM.

### Current Measurement and Solid State Relay

To protect the battery in case the BMS detects an error, a solid state relay consisting of two PMOS acting as a high side switch in a common source configuration are used. To allow current to flow in and out of the battery, the BMS needs to actively pull the Gates of the PFETs to GND. This is achieved by a logic level NFET connected to a GPIO on the STM.

In Series with the Solid State Relay is a 5mOhm Shunt resistor with a TSC2012 current amplifier. The Amplifier connects to the TLE9012DQU and is biased by a resistor divider to measure bidirectional currents. The reference is also measured by the TLE9012DQU so no additional compensation is required by the user. 

### Tips during assembly

Assembly is rather straightforward but I would strongly suggest using a stencil as well as lead free solder paste. Make sure to keep the amount of excessive solder paste as low as possible to prevent solder bridges und the Pins of the STM and TLE. While placing the components, make sure that components marked as DNP (do not populate) are not placed on the PCB, which may lead to unexpected behaviour.

After soldering the SMD components, it is highly recommended to check that a debug connection to the STM can be established using the ST-Link and a lab-bench power supply.

### Pinout Table

Pinout for J2 (Battery Connector):

| Pin number | Pin name | Pin net         |
|------------|----------|-----------------|
| 1          | Pin\_1   | Bat+            |
| 2          | Pin\_2   | Cell\_4+        |
| 3          | Pin\_3   | Cell\_3+        |
| 4          | Pin\_4   | Cell\_2+        |
| 5          | Pin\_5   | Cell\_1+        |
| 6          | Pin\_6   | Cell\_1-        |
| 7          | Pin\_7   | Bat-            |

Pinout for J4 (TMP1):

| Pin number | Pin name | Pin net                |
|------------|----------|------------------------|
| 1          | Pin\_1   | NTC+                   |
| 2          | Pin\_2   | NTC-                   |

Pinout for J3 (TMP2):

| Pin number | Pin name | Pin net                |
|------------|----------|------------------------|
| 1          | Pin\_1   | NTC+                   |
| 2          | Pin\_2   | NTC-                   |

Pinout for J1 (TMP3):

| Pin number | Pin name | Pin net                |
|------------|----------|------------------------|
| 1          | Pin\_1   | NTC+                   |
| 2          | Pin\_2   | NTC-                   |

Pinout for J9 (CPower Supply):

| Pin number | Pin name | Pin net         |
|------------|----------|-----------------|
| 1          | Pin\_1   | 5V              |
| 2          | Pin\_2   | GND             |

Pinout for J6 (RS485):

| Pin number | Pin name | Pin net         |
|------------|----------|-----------------|
| 1          | Pin\_1   | RS485-VCC (3.3-5V)    |
| 2          | Pin\_2   | A               |
| 3          | Pin\_3   | B               |
| 4          | Pin\_4   | RS485-GND       |

Pinout for J8 (UART\_Shell):

| Pin number | Pin name | Pin net         |
|------------|----------|-----------------|
| 1          | Pin\_1   | UART-GND        |
| 2          | Pin\_2   | TX              |
| 3          | Pin\_3   | RX              |
| 4          | Pin\_4   | UART-VCC (3.3-5V) |

Pinout for J10 (SW\_Debug\_Connector):

| Pin number | Pin name | Pin net        |
|------------|----------|----------------|
| 1          | Pin\_1   | +3V3           |
| 2          | Pin\_2   | SWCLK          |
| 3          | Pin\_3   | GND            |
| 4          | Pin\_4   | SWDIO          |
| 5          | Pin\_5   | NRST           |
| 6          | Pin\_6   | SWO            |

## Firmware Overview

As a result of the dual core nature of the STM32H755, the software consists of two separate sub-projects which share a common .ioc file for code generation and peripheral initilization, as well as a folder named Common, containing files present in both sub-projects. Both cores operate independently by default except a synchronization event at the start. Data between Cores is shared by a mailbox system further described in the Inter Processor Communication section.

To ensure the basic BMS functions mainly monitoring and communication of logdata to the host PC, these functions are running on the Cortex M4 core. The M4 core sends these data to the Cortex M7 which runs the State Estimation. This decoupling of monitoring and state estimation functions allows for testing new algorithms in real time while not compromising the monitoring function in case new state estimation algorithms are facing bugs or a too high computational complexity.

### Firmware Structure

The following graphic gives a brief overview about the firmware structure. Both Cores operate on a super loop approach so no overhead and complexity of a RTOS is introduced.

![](/readme_res/Software_Structure.drawio.png "Software structure overview")

Cortex M7:

In each loop iteration, the Cortex M7s main loop checks for updated measurement data from the M4. If new data is available, the state estimation code found in the BMS_State_Estimation/Src/BMS_State_Estimation.c is executed and the results can be passed back to the M4 via a Inter Processor Communication Mailbox. The OLED is then updated with new data in the Dusplay_Functions/Src/Display_Function.c file. The OLED is based on a SSD1306 controller IC which interfaces with the STM32 via I2C. The [stm32-ssd1306 library from Aleksander Alekseev](https://github.com/afiskon/stm32-ssd1306) is used to interface with the display.

To send debug messages over the isolated UART Interface, the Cortex M7s _write and __io_putchar prototypes are redefined to print data via UART in blocking mode. This allows for using the printf function of stdio.h to send data as ascii strings. Due to the blocking nature, debug messages will influence execution time of the main loop!

Cortex M4:

The Cortex M4 loop starts with executing the BMS_loop() function found in Core/Src/BMS_Functions.c. This loop function uses the abstraction layer found in the TLE9012_Lib folder to interface with the TLE9012 chip, initiates a measurement, serves the watchdog of the TLE, checks if any error occures and calls the BMS_Balance() function to handle cell balancing. Measurement Data is written into a global (in context of the Cortex M4) variable module which is of the shared datatype Cell_Module for further use.  The values of the module variable are then passed to the Cortex M7 via an Inter Processor Communication Mailbox. 

After evaluating the state of the BMS, the rs485_communication_loop() function handling the serial RS485 protocoll is called. The RS485 code is written in a way to prevent the communication protocol from interfering with the cell monitoring by the use of receive and transmit interrupts. In the rs485_communication_loop() it is first checked if a packet is received successfully or a timeout occured. If a packet was successfully received, the rs485_process_package() function is called processing the received packet and sending an appropriate response. The Transmit is also interrupt based to ensure continous monitoring of the battery.

### Inter Processor Communciation

Both STM32 Cores can work fully independently with their own peripherals, RAM and Flash sections. To interact with each other, a mechanism has to be established. Their are many different approaches to achive this. The main methods are Hardware Semaphores and Shared Memory. Shared Memory is a region in SRAM accessed by both cores to exchange data. Hardware Semaphores are a peripheral that can be used for management of shared ressources and to pass event in form of interrupts between cores. ST suggests using the OpenAMP Framework as a middleware to handle the transfer of data through these mechanisms. 

While the OpenAMP library provides a great set of tools to reliably solve this problem, a much simpler approach was used in this project which can be found in the Common/Inter_Process_Communication folder. SRAM4 is used as shared memory as suggested in application notes of STMicro. A Number of mailboxes is initialized by the Cortex M7 before the start of the Cortex M4 each containing a header consisting of a flag to signalize if new data is available, a flag to protect the mailbox during read and write accesses, a length value and the data itself. A mailbox register table is initilized at the start of SRAM4 to keep track of mailbox addresses and states. Access to mailboxes is provided by read and write methods. To check the status of a mailbox, a check function is provided. 

To ensure correct interpretation of datatypes exchanged via mailboxes, a Shared_DataTypes header is included in the common folder, ensuring access to the datatype in both the M4 and M7 program files.

### Serial Protocol

The Serial Protocol used in the RS485 interface at (115200 baud 8N1) is uses the following format

| Byte 1 | Byte 2 | Byte 3 | n Bytes  | Byte n+1 |
| ------ | ------ | ------ | -------- | ------ |
| Bus ID | Command | Payload Length n | Payload | Checksum |

The Bus ID of the device is hardcoded in the Macro RS485_ID found in RS485_Communication.h and needs to be unique for each BMS on the same bus. The BMS respondes to commands using the same format, addressing the host with the ID 0x80. In case the payload length is zero, byte 3 is directly followed by the checksum. The checksum is currently not implemented and therefore can be ignored in processing but has to be transmitted.

Currently implemented commands are GET_MODULE_INFO which returns module info in a binary format as well as GET_MODULE_INFO_AS_STRING which returns the same data in a ascii format which is human readable at the expanse of taking more bus bandwidth.

# Extending the Firmware

The firmware in this repository builds a foundation to experiment with more complex BMS algorithms. Therefore it tries to provide easy expansion/callbacks for the following use cases: 

## State estimation

The bmsStateEstimationCallback is called everytime there are updated measurements from the cortex M4 and returns new estimations about the state of each cell. State estimation algorithms shall therefore be added in this function.

## Balancing

Balancing is currently seen as a Cortex M4 function and is part of the BMS_Functions.c. The implementation is a simple Top-Balancing algorithm based on cell voltage and called from the BMS_Loop. A proper callback function inside the Cortex M7s codebase might be implemented in the future and this documentation will be updated .

## Communication Protocol

New commands for the RS485 interface might be added in the rs485_process_package function by appending cases to the switch(command) construct. It is strongly recommended to use 1. a preprocessor macro to assign the command ID a human readable name and 2. use a dedicated function called from the switch case construct to keep the readability of the function high. It is also strongly recommended to follow the naming scheme rs485_command_yourFunctionName()

# Known quirks and errors

### Single wire UART on STM32 is too slow

Because the TLE9012DQU uses a single wire based UART protocol, the basic idea was to use the single wire UART mode of the STM as well. Turns out, that the switch between TX and RX is way to slow resulting in missed bytes when reading data from the TLE9012. To prevent this, UART4 RX was connected on a spare pin with the single wire uart from the STM now exclusivly running in TX mode. This should be implemented in the Layout in further PCB revisions

### STM32 doesn't connect to ST-Link after firmware update

If the STM32 doesn't connect to the ST-Link after a Firmware update, the new version of the Firmware might activate the onboard switching regulator option of the STM32. Because no switching regulator is used in this design, ensure that the LDO option is activted by software instead. To recover the chip, use a jumper to connect the STM32 side of R41 with 3V3 during a power cycle. The STM32 will then start in the internal bootloader instead of the user software, allowing for erasing the flash/reprogramming the device

### Display was hotplugged and is not powering on anymore

This cheap OLEDs are not hotpluggable and might be dead after such an event. Solution is to buy another display.

# TODO

- Individuelle BMS Kommandos dokumentieren -> ggf Beispiele