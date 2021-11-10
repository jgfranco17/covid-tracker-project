import sys
import os
import traceback as tb
from analysis import CovidData
from general import Colors, timestamp


class Controller(object):
    def __init__(self, model:CovidData):
        self.running = True
        self.object = model
        self.options = [
            "[0] Exit program",
            "[1] Display tracker properties",
            "[2] Update data",
            "[3] Display history",
            "[4] Check status"
        ]

        self.input_history = []

        print(Colors.yellow(f"\n[{timestamp()}]"))
        print(f"Controller online.")

    def run(self):
        """
        Looped control function, used to continuously run application.
        """
        while self.running:
            try:
                print('OPTIONS:')
                print("\n".join(self.options))
                user_choice = int(input("\nEnter choice: "))

                # Enact user choice
                if user_choice == 0:
                    print(Colors.red("[SHUTTING DOWN]"))
                    self.running = False
                    sys.exit()
                elif user_choice == 1:
                    print(repr(self.object))
                elif user_choice == 2:
                    self.object.update()
                elif user_choice == 3:
                    print("|".join(self.input_history))
                elif user_choice == 4:
                    print(f"Running: {self.running}")
                else:
                    print("Invalid input!")

                # Ensures a maximum capacity of 30 history items
                if len(self.input_history) == 30:
                    del self.input_history[0]
                self.input_history.append(str(user_choice))
                print("")

            except Exception as e:
                print(f'{Colors.red("ERROR - Controller failed.")}\nReason: {e}')
                tb.print_exc()
