from Tool.WindowsAPI import key_check
import random


# map user input into action space


class User:
    def __init__(self):
        self.D = 1
        self.DOWN = False
        self.UP = False

    def get_user_action(self):
        operation, direction = key_check()
        for d in direction:
            if d == 'A':
                self.D = 0
            elif d == 'D':
                self.D = 1
            elif d == 'W':
                self.UP = True
            elif d == 'S':
                self.DOWN = True

        for op in operation:
            if op == 'L':
                return random.randint(6, 8)
            elif op == 'J':
                if self.UP:
                    self.UP = False
                    return 5
                else:
                    if self.D == 0:  # left
                        return 3
                    elif self.D == 1:
                        return 4
            elif op == 'K':
                if self.UP:
                    self.UP = False
                    return 11
                elif self.DOWN:
                    self.DOWN = False
                    return 12
                else:
                    if self.D == 0:  # left
                        return 9
                    elif self.D == 1:
                        return 10
            elif op == 'Shift':
                if self.D == 0:
                    return 13
                elif self.D == 1:
                    return 14

        if 'A' in direction:
            return 1
        elif 'D' in direction:
            return 2

        return 0
