from practice import PracticeClass

prac = PracticeClass()

print("value of function f:", prac.f(1))

print("value of function Fs:", prac.Fs(x=2))


print("check callable, ", callable(prac.Fs), callable(prac.Fss))


class gradient:

    def fx(self, f, x):

        return f(x)



print("gradient of function Fs:", gradient().fx(prac.Fs, 1))



print("value of fx:", gradient().fx(prac.Fss(), 1))
