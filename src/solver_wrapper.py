import struct
from enum import Enum
import collections
import importlib

#  Import Z3 if module is installed
z3_module = importlib.util.find_spec("z3")
z3_found = z3_module is not None
# print(f'Z3 found? {z3_found}')
if z3_found:
    z3 = importlib.import_module("z3")
    # unclear if necessary to call importlib.invalidate_caches() and add a module to sys.modules
    # for more info check https://docs.python.org/3/library/importlib.html#importlib.import_module

#  Import CVC4 if the module exists
cvc4_module = importlib.util.find_spec("pycvc4")
cvc4_found = cvc4_module is not None
# print(f'CVC4 found? {cvc4_found}')
if cvc4_found:
    pycvc4 = importlib.import_module("pycvc4")
    kinds = importlib.import_module("..kinds", package="pycvc4.subpkg")


# TODO mathsat


class Result(Enum):
    SAT = 1
    UNSAT = 2
    UNKNOWN = 3


class Solver:

    def __init__(self, solverName='CVC4', fpBitExp=8, fpBitS=24):
        self.name = solverName.upper()
        self.bitExp = fpBitExp
        self.bitSig = fpBitS

        if solverName.upper() == 'Z3':
            self.slv = z3.Solver()
            self.fp = z3.FPSort(fpBitExp, fpBitS)
            self.rm = z3.RNE()
            z3.set_option(timeout=1000)  # 16.7 minutes
        # elif solverName.capitalize == 'MATHSAT':
        #     self.slv = ???
        else:
            self.slv = pycvc4.Solver()
            self.slv.setOption("produce-models", "true")
            self.slv.setLogic("FP")

            self.fp = self.slv.mkFloatingPointSort(fpBitExp, fpBitS)
            self.rm = self.slv.mkRoundingMode(pycvc4.RoundNearestTiesToEven)

    def getName(self):
        return self.name

    def add(self, *queries):
        if self.name == 'Z3':
            self.slv.add(queries)
        else:
            for q in queries:
                self.slv.assertFormula(q)

    def reset(self):
        if self.name == 'Z3':
            self.slv.reset()
        else:
            self.slv.resetAssertions()

    def set(self, param_name, val):
        if self.name == 'Z3':
            self.slv.set(param_name, val)
        else:
            self.slv.setOption(param_name, val)

    def check(self):
        if self.name == 'Z3':
            res = self.slv.check()
            if res == z3.sat:
                return Result.SAT
            elif res == z3.unsat:
                return Result.UNSAT
            else:
                return Result.UNKNOWN
        else:  # CVC4
            res = str(self.slv.checkSat())  # temporary workaround ! TODO: fix this when API is fixed
            if res == 'sat':  # weirdly res.isSat() is not a part of Python API (but there in C)
                return Result.SAT
            elif res == 'unsat':  # res.isUnsat():
                return Result.UNSAT
            else:  # isSatUnknown()
                return Result.UNKNOWN

    def to_smt(self):
        return self.slv.to_smt2()

    def getValue(self, t):
        """
        Returns floating-point value of the variable [t] from the model returned by SMT solver
        :param t: term - variable
        :return: floating-point value of the variable
        """
        if self.name == 'Z3':
            model = self.slv.model()
            declNames = [d.name() for d in model.decls()]
            if str(t) in declNames:
                tmp = model[t]
                if z3.is_real(tmp):
                    # print(f'model type: {type(tmp)}')
                    # print(float(tmp.as_decimal(16)))
                    return float(tmp.as_decimal(16).replace('?', ''))
                elif tmp.isNaN():
                    raise Exception('Z3 returned NaN despite the constraints')
                elif tmp.isZero():
                    return 0
                else:
                    significand = tmp.significand()
                    exponent = tmp.exponent()
                    value = -1 * float(significand) * pow(2, float(exponent) - 127) if tmp.sign() else float(
                        significand) * pow(2, float(exponent) - 127)
                    # print('VALUE: %s = %s' % (t, value))
                    return value
            else:
                raise Exception("Unknown term")
        else:  # CVC4
            binary = self.slv.getValue(t)
            # get the pure bits
            bits = str(binary).replace('(fp', '').replace(' #b', '').replace(')', '')
            intRepr = int(bits, 2)
            return struct.unpack('f', struct.pack('I', intRepr))[0]

    def FPVal(self, v):
        if self.name == 'Z3':
            return z3.FPVal(v, self.fp)
        else:
            bitv = self.slv.mkBitVector(binary(v), 2)
            return self.slv.mkFloatingPoint(self.bitExp, self.bitSig, bitv)

    def FP(self, varName):  # add tpe as a third parameter if we want mixed types
        if self.name == 'Z3':
            return z3.FP(varName, self.fp)
        else:
            return self.slv.mkConst(self.fp, varName)

    # def fpIsInfinite(self, x):
    #     if self.name == 'Z3':
    #         return z3.fpIsInf(x)
    #     else:
    #         return self.slv.mkConst(self.fp, varName)

    def fpNaN(self):
        if self.name == 'Z3':
            return z3.fpNaN(self.fp)
        else:
            return self.slv.mkNaN(self.bitExp, self.bitSig)

    def fpInf(self, negative):
        """
        Returns floating-point positive or negative infinity
        :param negative: Boolean - True stands for negative infinity -oo, False for positive +oo
        :return:
        """
        if self.name == 'Z3':
            return z3.fpInfinity(self.fp, negative)
        else:
            if negative:
                return self.slv.mkNegInf(self.bitExp, self.bitSig)
            else:
                return self.slv.mkPosInf(self.bitExp, self.bitSig)

    # Operations
    def And(self, *args):
        if self.name == 'Z3':
            flatArgs = args[0] if len(args) == 1 and isinstance(args[0], collections.abc.Iterable) else args
            return z3.And(flatArgs)
        else:
            count = len(args)
            if count <= 1 and not isinstance(args, collections.abc.Iterable):
                raise Exception("Missing arguments for And(a,b,...)")
            flatArgs = [item for sublist in args for item in sublist] if count == 1 else args
            tmpterm = flatArgs[0]
            i = 1
            while i < len(flatArgs):
                tmpterm = self.slv.mkTerm(kinds.And, tmpterm, flatArgs[i])
                i = i + 1
            return tmpterm

    def Or(self, *args):
        if self.name == 'Z3':
            flatArgs = args[0] if len(args) == 1 and isinstance(args[0], collections.abc.Iterable) else args
            return z3.Or(flatArgs)
        else:
            count = len(args)
            if count <= 1 and not isinstance(args, collections.abc.Iterable):
                raise Exception("Missing arguments for Or(a,b,...)")
            flatArgs = [item for sublist in args for item in sublist] if count == 1 else args
            tmpterm = flatArgs[0]
            i = 1
            while i < count:
                tmpterm = self.slv.mkTerm(kinds.Or, tmpterm, flatArgs[i])
                i = i + 1
            return tmpterm

    def Not(self, a):
        if self.name == 'Z3':
            return z3.Not(a)
        else:
            return self.slv.mkTerm(kinds.Not, a)

    def fpLEQ(self, x, y):
        if self.name == 'Z3':
            return z3.fpLEQ(x, y)
        else:
            return self.slv.mkTerm(kinds.FPLeq, x, y)

    def fpEQ(self, x, y):
        if self.name == 'Z3':
            return z3.fpEQ(x, y)
        else:
            return self.slv.mkTerm(kinds.FPEq, x, y)

    def fpNEQ(self, x, y):
        if self.name == 'Z3':
            return z3.fpNEQ(x, y)
        else:
            return self.slv.mkTerm(kinds.Not, self.slv.mkTerm(kinds.FPEq, x, y))

    def fpAdd(self, x, y):
        if self.name == 'Z3':
            return z3.fpAdd(self.rm, x, y)
        else:
            return self.slv.mkTerm(kinds.FPPlus, self.rm, x, y)

    def fpSub(self, x, y):
        if self.name == 'Z3':
            return z3.fpSub(self.rm, x, y)
        else:
            return self.slv.mkTerm(kinds.FPSub, self.rm, x, y)

    def fpMul(self, x, y):
        if self.name == 'Z3':
            return z3.fpMul(self.rm, x, y)
        else:
            return self.slv.mkTerm(kinds.FPMult, self.rm, x, y)

    def fpDiv(self, x, y):
        if self.name == 'Z3':
            return z3.fpDiv(self.rm, x, y)
        else:
            return self.slv.mkTerm(kinds.FPDiv, self.rm, x, y)

    def fpNeg(self, x):
        if self.name == 'Z3':
            return z3.fpNeg(x)
        else:
            return self.slv.mkTerm(kinds.FPNeg, x)

    # Real valued functions
    def Real(self, varName):
        if self.name == 'Z3':
            return z3.Real(varName)
        else:
            return None  # self.slv.mkConst(self.fp, varName)

    def RealVal(self, v):
        if self.name == 'Z3':
            return z3.RealVal(v)  # TODO: figure out whether to give value as string
        else:
            return None

    def realLEQ(self, x, y):
        if self.name == 'Z3':
            return x <= y
        else:
            return None

    def realLE(self, x, y):
        if self.name == 'Z3':
            return x < y
        else:
            return None

    def realEQ(self, x, y):
        if self.name == 'Z3':
            return x == y
        else:
            return None

    def realAdd(self, x, y):
        if self.name == 'Z3':
            return x + y
        else:
            return None

    def realSub(self, x, y):
        if self.name == 'Z3':
            return x - y
        else:
            return None

    def realMul(self, x, y):
        if self.name == 'Z3':
            return x * y
        else:
            return None

    def realDiv(self, x, y):
        if self.name == 'Z3':
            return x / y
        else:
            return None

    def realNeg(self, x):
        if self.name == 'Z3':
            return - x
        else:
            return None

    def realIf(self, cond, then, elze):
        if self.name == 'Z3':
            return z3.If(cond, then, elze)
        else:
            return None


# from https://stackoverflow.com/questions/16444726/binary-representation-of-float-in-python-bits-not-hex
def binary(num):
    return ''.join(bin(c).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', num))
