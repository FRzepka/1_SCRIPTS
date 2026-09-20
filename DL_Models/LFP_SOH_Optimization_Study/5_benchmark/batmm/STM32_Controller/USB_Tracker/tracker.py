# Big help from https://forum.avhzy.com/forum.php?mod=viewthread&tid=369&highlight=linux

from serial import Serial
from struct import unpack as unpack
from struct import pack_into as pack_into
from argparse import ArgumentParser, FileType
from time import time, sleep, gmtime, localtime, mktime
from sys import exit

# Actions along with their description, no list function
__actions = ["read"]
__action_descriptions = {
    "read": "Reads energy value from the meter"
}

# Values for read operation
__gets = ["all", "voltage (V)", "current (A)", "power (W)", "voltageDP (V)", "voltageDM (V)", "energy (Wh)"]

class AVHzY_CT3:
    def __init__(self, device, action, repeat, time, reads, output):
        self.__action = action
        self.__repeat = repeat
        self.__time = time
        self.__output = output
        self.__ser = Serial(device)

        self.__first_exec = True
        self.__timestamp = 0
        self.__energy = 0
        self.__prev_timestamp = 0

        if reads == "all":
            self.__reads = ["voltage (V)", "current (A)", "power (W)", "voltageDP (V)", "voltageDM (V)", "energy (Wh)"]
        else:
            self.__reads = reads

        # Action handlers in a dictionary
        self.__action_handlers = {
            "list": self.__action_list,
            "read": self.__action_read,
        }

    def __del__(self):
        self.__ser.close()
        self.__output.close()

    def __action_list(self):
        print("Actions list:")
        for action in __action_descriptions:
            print("  ", action, "\t", __action_descriptions[action])

    def xor_checksum(self, byte_array, start, end):
        checksum = 0
        for byte in byte_array[start : end + 1]:
            checksum ^= byte
        return checksum

    def __action_read(self):
        if self.__first_exec:
            self.__output.write("time")
            for r in self.__reads:
                self.__output.write(",{0}".format(r))
            self.__output.write("\n")
            self.__first_exec = False

        self.__ser.flushInput()
        while True:
            packet = self.__ser.read(30)
            if len(packet) == 30:
                break

        packetData = unpack("<fffff", packet[9:29])  # Byte might vary on a different USB meter

        voltage = packetData[0]
        current = packetData[1]
        power = packetData[2]
        voltageDP = packetData[3]
        voltageDM = packetData[4]
        self.__timestamp = int(time()*1000)

        self.__output.write("{0}".format(self.__timestamp))
        if "voltage (V)" in self.__reads:
            self.__output.write(",{0:.6f}".format(voltage))
        if "current (A)" in self.__reads:
            self.__output.write(",{0:.6f}".format(current))
        if "power (W)" in self.__reads:
            self.__output.write(",{0:.6f}".format(power))
        if "voltageDP (V)" in self.__reads:
            self.__output.write(",{0:.6f}".format(voltageDP))
        if "voltageDM (V)" in self.__reads:
            self.__output.write(",{0:.6f}".format(voltageDM))

        if "energy (Wh)" in self.__reads:
            if self.__prev_timestamp == 0:
                self.__output.write(",0")
            else:
                self.__energy += ((self.__timestamp - self.__prev_timestamp) * power) / 3600000000
                self.__output.write(",{0:.6f}".format(self.__energy))

        self.__output.write("\n")
        self.__output.flush()

        self.__prev_timestamp = self.__timestamp

    def perform_action(self):

        self.__ser.reset_input_buffer()

        resetCommand = bytearray.fromhex("A5 04 00 00 00 01 07 00 00 06 5A")
        self.__ser.write(resetCommand)
        self.__ser.read(11)

        resetRecordTimerCommand = bytearray.fromhex("A5 04 00 00 00 01 0C 0A 00 07 5A")
        self.__ser.write(resetRecordTimerCommand)
        self.__ser.read(11)

        startReadCommand = bytearray.fromhex(
            "A5 08 00 00 00 01 09 0B 00 00 00 00 00 00 5A"
        )
        pack_into("I", startReadCommand, 9, self.__time)
        startReadCommand[13] = self.xor_checksum(startReadCommand, 5, 12)
        self.__ser.write(startReadCommand)
        self.__ser.read(11)

        count = 0
        while count != self.__repeat:
            try:
                self.__action_handlers[self.__action]()
            except KeyboardInterrupt:
                break

            if self.__repeat != -1:
                count += 1
            sleep(self.__time / 1000.0)

    def change_output(self, output):
        self.__output.close()
        self.__output = output


def main():
    parser = ArgumentParser(description="Program to interact with the AVHzY CT-3 power meter")
    parser.add_argument("action", metavar="action", choices=__actions, help="The action to perform [choices: %(choices)s]")
    parser.add_argument("-d", "--device", default="/dev/ttyACM1", help="Path to the device [default: %(default)s]")
    parser.add_argument("-r", "--repeat", type=int, default=-1, help="How many times to repeat the operation. Must be in [-1, inf[ (-1: infinite) [default: %(default)s]")
    parser.add_argument("-t", "--time", type=int, default=100, help="The time (in milliseconds) to wait between each action iteration.")
    parser.add_argument("-o", "--output", default="-", type=FileType("w"), help="Where to write output of the action [default: stdout]")
    parser.add_argument("-g", "--get", default="all", choices=__gets, nargs="+", help="For read operation: what to get from power meter [choices: %(choices)s default: %(default)s")
    args = parser.parse_args()

    if args.repeat < -1:
        print("ERROR: repeat must be in >= -1")
        parser.print_usage()
        exit(1)

    if args.time < 1:
        print("ERROR: time must be > 1")
        parser.print_usage()
        exit(1)

    AVHzY_CT3(args.device, args.action, args.repeat, args.time, args.get, args.output).perform_action()
    exit(0)

if __name__ == '__main__':
    main()
