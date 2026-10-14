class LFSR:
    def __init__(self, seed=0xACE1):
        self.state = seed & 0xFFFFFFFF
        if self.state == 0:
            self.state = 1

    def step(self):
        # Полином №12: X^32 + X^29 + X^28 + X^23 + X^22 + X^21 + X^16 + X^13 + X^10 + X^6 + X^5 + X^4 + 1
        # Степени: 32, 29, 28, 23, 22, 21, 16, 13, 10, 6, 5, 4 (сдвиги 31, 28, 27, 22, 21, 20, 15, 12, 9, 5, 4, 3)
        bit = (
            (self.state >> 31) ^
            (self.state >> 28) ^
            (self.state >> 27) ^
            (self.state >> 22) ^
            (self.state >> 21) ^
            (self.state >> 20) ^
            (self.state >> 15) ^
            (self.state >> 12) ^
            (self.state >> 9)  ^
            (self.state >> 5)  ^
            (self.state >> 4)  ^
            (self.state >> 3)
        ) & 1

        self.state = ((self.state << 1) | bit) & 0xFFFFFFFF
        return self.state

    def random(self):
        # Генерация 32 бит для получения случайного числа float в диапазоне [0, 1)
        for _ in range(32):
            self.step()
        return self.state / 0xFFFFFFFF