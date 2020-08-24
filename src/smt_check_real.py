import ast
import time

import shape_utils
from solver_wrapper import Result


class SMTVisitorReal(ast.NodeVisitor):
    def __init__(self, solver, vars, primed):
        super().__init__()
        self.primed = primed
        self.s = solver
        self.smt = solver.Real(0)
        self.vars = vars

    def visit_BinOp(self, node):
        """ Define a Real operation in SMT query for a binary operation in the loop. """
        if isinstance(node.op, ast.Add):
            smt = self.s.realAdd(self.visit(node.left), self.visit(node.right))
        elif isinstance(node.op, ast.Mult):
            smt = self.s.realMul(self.visit(node.left), self.visit(node.right))
        elif isinstance(node.op, ast.Sub):
            smt = self.s.realSub(self.visit(node.left), self.visit(node.right))
        elif isinstance(node.op, ast.Div):
            smt = self.s.realDiv(self.visit(node.left), self.visit(node.right))
        self.smt = smt
        return smt

    def visit_BoolOp(self, node):
        # boolean operations
        if isinstance(node.op, ast.And):
            operands = [self.visit(x) for x in node.values]
            smt = self.s.And(operands)
        elif isinstance(node.op, ast.Or):
            smt = self.s.Or([self.visit(x) for x in node.values])
        self.smt = smt
        return smt

    def visit_Compare(self, node):
        if isinstance(node.ops[0], ast.LtE):
            smt = self.s.realLEQ(self.visit(node.left), self.visit(node.comparators[0]))
        elif isinstance(node.ops[0], ast.Lt):
            smt = self.s.realLE(self.visit(node.left), self.visit(node.comparators[0]))
        elif isinstance(node.ops[0], ast.GtE):
            smt = self.s.realLEQ(self.visit(node.comparators[0]), self.visit(node.left))
        elif isinstance(node.ops[0], ast.Gt):
            smt = self.s.realLE(self.visit(node.comparators[0]), self.visit(node.left))
        elif isinstance(node.ops[0], ast.Eq):
            smt = self.s.realEQ(self.visit(node.left), self.visit(node.comparators[0]))
        self.smt = smt
        return smt

    def visit_UnaryOp(self, node):
        """ Define a Real operation in SMT query for a unary minus and negation operation in the loop. """
        if isinstance(node.op, ast.USub):
            smt = self.s.realNeg(self.visit(node.operand))
        elif isinstance(node.op, ast.Not):
            smt = self.s.Not(self.visit(node.operand))
        else:
            raise Exception("Unary operators except negation are not defined")
        self.smt = smt
        return smt

    def visit_Num(self, node):
        """ Define a Real operation in SMT query for a constant in the loop. """
        smt = self.s.RealVal(node.n)
        self.smt = smt
        return smt

    def visit_Name(self, node):
        """
        Define a FP operation in SMT query for a variable in the loop.
        If a variable has been assigned in a loop already, it will be primed.
        """
        smt = self.vars[str(node.id) + '!'] if node.id in self.primed else self.vars[node.id]
        self.smt = smt
        return smt


def getLoopBody(s, loop, mapVars):
    """ Forms SMT query for the transfer function of the loop """
    # TODO define a correct order of statements in a loop (include line numbers?)
    body = []
    assignedVars = list()

    for branch in loop:
        ifStmts = []
        elseStmts = []

        if 'not' in branch:
            continue

        for v in loop[branch]:
            xPrime = mapVars[str(v) + '!']
            tree = ast.parse(loop[branch][v])
            visitor = SMTVisitorReal(s, mapVars, [])  # when we can use diff statements, assignedVars)
            visitor.visit(tree)
            rhs = visitor.smt
            ifStmts.append(s.realEQ(xPrime, rhs))
            assignedVars.append(v)

        if branch != 'true':
            tree = ast.parse(branch)
            visitor = SMTVisitorReal(s, mapVars, [])
            visitor.visit(tree)
            smtThenCondition = visitor.smt

            assignedVars = []
            # # TODO take care of nested conditions
            # condList = branch.split(' and ')
            # for i,c in enumerate(condList):
            elseCond = 'not ' + branch
            for v in loop[elseCond]:
                xPrime = mapVars[str(v) + '!']
                tree = ast.parse(loop[elseCond][v])
                visitor = SMTVisitorReal(s, mapVars, [])  # when we can use diff statements, assignedVars)
                visitor.visit(tree)
                rhs = visitor.smt
                elseStmts.append(s.realEQ(xPrime, rhs))
                assignedVars.append(v)
            body.append(s.realIf(smtThenCondition, s.And(ifStmts), s.And(elseStmts)))
        else:
            body.append(s.And(ifStmts))

    return s.And(body)


# nondet is a map of non-deterministic variable to its range
def getInvariant(s, vars, outRanges, mapVars, nondet, shapeCoefs, coefsToVars, assumptions, primed=False):
    def addTerms(acc, tlist):
        if len(tlist) == 0:
            return acc
        else:
            return addTerms(s.realAdd(tlist[len(tlist) - 1], acc), tlist[:len(tlist) - 1])

    # encode all ranges for invariant
    rangeInvariant = list()
    for v in outRanges:
        (lo, up) = outRanges[v]
        x = mapVars[str(v) + '!'] if primed else mapVars[str(v)]
        (lo, up) = (s.RealVal(float(lo)), s.RealVal(float(up)))
        rangeInvariant.append(s.realLEQ(lo, x))
        rangeInvariant.append(s.realLEQ(x, up))

    if not primed:
        for v in nondet:
            (lo, up) = nondet[v]
            x = mapVars[str(v)]
            (lo, up) = (s.RealVal(float(lo)), s.RealVal(float(up)))
            rangeInvariant.append(s.realLEQ(lo, x))
            rangeInvariant.append(s.realLEQ(x, up))

    
    terms = []
    for i, c in enumerate(shapeCoefs[1:], 1):
        if not c == 0.0:
            # c * x * x * y
            term = s.RealVal(c)

            # multiply with all factors
            vlist = coefsToVars[i].rsplit('*')
            for vIndex in vlist:
                vName = f'{vars[int(vIndex)]}!' if primed else str(vars[int(vIndex)])
                smtVar = mapVars[vName]
                term = s.realMul(term, smtVar)
            terms.append(term)

    preshape = addTerms(terms[0], terms[1:])
    bound = s.RealVal(-shapeCoefs[0])
    shape = s.realLEQ(preshape, bound)

    # add the assumed correlation between vars for bounded loops
    smtAssumptions = []
    if not primed:
        for assume in assumptions:
            tree = ast.parse(assume)
            visitor = SMTVisitorReal(s, mapVars, [])
            visitor.visit(tree)
            smtAssumptions.append(visitor.smt)

    rangesI = s.And(rangeInvariant)
    return s.And(rangesI, shape, s.And(smtAssumptions))


def getInitSMTQuery(s, program, mapVars, nondet, outRanges, shapeCoefs, vars, coefsToVars):
    # encode all ranges for precondition
    inputRanges = program['inputRanges']
    rangePre = list()
    for v in inputRanges:
        (lo, up) = inputRanges[v]
        x = mapVars[v]
        # TODO convert to some rational number?
        (lo, up) = (s.RealVal(float(lo)), s.RealVal(float(up)))
        rangePre.append(s.realLEQ(lo, x))
        rangePre.append(s.realLEQ(x, up))

    # check that precondition implies the invariant
    # pre and (not inv) == unsat?
    precondition = s.And(rangePre)
    # s, vars, outRanges, mapVars, shapeCoefs, coefsToVars,
    invariant = getInvariant(s, vars, outRanges, mapVars, nondet, shapeCoefs, coefsToVars, program['assume'],  primed=False)
    init = s.And(precondition, s.Not(invariant))
    # print(simplify(init))
    return init


def getInductiveStepSMTQuery(s, program, mapVars, nondet, outRanges, shapeCoefs, coefsToVars):
    vars = program['vars']

    # check that invariant hold after k-th iteration
    # inv and loop and (not inv') == unsat?
    invariant = getInvariant(s, vars, outRanges, mapVars, nondet, shapeCoefs, coefsToVars, program['assume'], primed=False)
    invP = getInvariant(s, vars, outRanges, mapVars, nondet, shapeCoefs, coefsToVars, program['assume'], primed=True)
    loop = getLoopBody(s, program['loopBody'], mapVars)
    transf = s.And(invariant, loop, s.Not(invP))

    return transf


def checkInductiveWOCex(s, vars, mapVars, nondet, init, inductiveStep, cexs, distFactor, outRanges={}, debug=False):
    """
    Checks inductiveness while blocking the interval with radius dist around cex's
    :param s:
    :param vars:
    :param mapVars:
    :param init:
    :param inductiveStep:
    :param cexs:
    :param dist:
    :return:
    """
    global blocked
    # prepare the solver for the new query
    s.reset()
    blockCex = []
    # block counterexamples
    for cex in cexs:
        blockOneCex = []
        # print(f'cex: {cex}')
        for v in vars:
            lo, up = outRanges[v]
            dist = abs(up - lo)*distFactor
            value = cex[v]
            x = mapVars[v]
            # print(f'{x} not in {(value-dist)} and {(value+dist)}')
            (lo, hi) = (s.RealVal(round(value - dist, 2)), s.RealVal(round(value + dist, 2)))
            blockOneCex.append(s.And(s.realLEQ(lo, x), s.realLEQ(x, hi)))
        blockCex.append(s.Not(s.And(blockOneCex)))

    if cexs:
        blocked = s.And(blockCex)

    # check that precondition implies the invariant
    # pre and (not inv) == unsat?
    s.add(init)
    if cexs:
        s.add(blocked)
    resInit = s.check()
    cex = {}
    if resInit == Result.SAT:
        for v in (vars + list(nondet.keys())):
            cex[v] = s.getValue(mapVars[v])
        return cex, True
    elif resInit == Result.UNKNOWN:
        timestamp = time.time_ns()
        f = open(f'query_init{timestamp}.sl', 'w')
        q = s.to_smt()
        f.write(q)
        f.close()
        raise TimeOutException(f'Timeout init query, see "query_init{timestamp}.sl"')
    s.reset()
    # check that invariant hold after k-th iteration
    # inv and loop and (not inv') == unsat?
    s.add(inductiveStep)
    if cexs:
        s.add(blocked)
    t0 = time.time()
    result = s.check()
    t1 = time.time()
    if debug:
        print(f'time inductive: {t1 - t0}')
    cex = {}
    if result == Result.SAT:
        for v in (vars + list(nondet.keys())):
            cex[v] = s.getValue(mapVars[v])
        # for v in vars:
        #     cex[v + '!'] = s.getValue(mapVars[v + '!'])
    elif resInit == Result.UNKNOWN:
        timestamp = time.time_ns()
        f = open(f'query_ind{timestamp}.sl', 'w')
        q = s.to_smt()
        f.write(q)
        f.close()
        raise TimeOutException(f'Timeout init query, see "query_ind{timestamp}.sl"')
    return cex, False


def checkInvariantWithCex(s, program, outRanges, shapeCoefs, vars, coefsToVars, numCex, dist, debug=False):
    """
    :return: list of counterexamples, whether one of the cex fails the initial condition
    """
    mapVars = {v: s.Real(str(v)) for v in (vars + program['nondet'])}
    primed = {str(v) + '!': s.Real(str(v) + '!') for v in vars}
    mapVars.update(primed)

    nondet = {v:(float(program['inputRanges'][v][0]), float(program['inputRanges'][v][1])) for v in program['nondet']}

    initQuery = getInitSMTQuery(s, program, mapVars, nondet, outRanges, shapeCoefs, vars, coefsToVars)
    indStepQuery = getInductiveStepSMTQuery(s, program, mapVars, nondet, outRanges, shapeCoefs, coefsToVars)

    # print(f'current time {time.asctime()}')
    # if the query times out, pass a timeout exception to the caller
    large_timeout = 1000 # 16.7 minutes
    s.set('timeout', large_timeout)
    cex, initCondFailed = checkInductiveWOCex(s, vars, mapVars, nondet, initQuery, indStepQuery, [], dist, debug)

    if not cex:
        return [], initCondFailed

    cexs = [cex]
    initConditionFailed = initCondFailed
    i = 0

    while i < numCex:
        s.reset()
        i = i + 1
        # s, vars, mapVars, init, inductiveStep, cexs, dist
        timeout = 10  # 10 seconds timeout
        try:
            s.set('timeout', timeout)
            new_cex, initCondFailed = checkInductiveWOCex(s, vars, mapVars, nondet, initQuery, indStepQuery, cexs, dist, outRanges, debug)
        except TimeOutException:
            return cexs, initConditionFailed

        if not new_cex or new_cex in cexs:  # can happen for symmetric points
            return cexs, initConditionFailed  # no new cex found, return what we have
        else:
            cexs.append(new_cex)
            initConditionFailed = initConditionFailed or initCondFailed 

    return cexs, initConditionFailed


def checkInvariantWithSymPts(s, program, outRanges, shapeCoefs, vars, coefsToVars, numCex, dist, debug=False):
    """
    :return: list of counterexamples, whether one of the cex fails the initial condition
    """
    mapVars = {v: s.Real(str(v)) for v in (vars + program['nondet'])}
    primed = {str(v) + '!': s.Real(str(v) + '!') for v in vars}
    mapVars.update(primed)

    nondet = {v:(float(program['nondet'][v][0]), float(program['nondet'][v][1])) for v in program['nondet']}

    initQuery = getInitSMTQuery(s, program, mapVars, nondet, outRanges, shapeCoefs, vars, coefsToVars)
    indStepQuery = getInductiveStepSMTQuery(s, program, mapVars, nondet, outRanges, shapeCoefs, coefsToVars)

    cex, initCondFailed = checkInductiveWOCex(s, vars, mapVars, nondet, initQuery, indStepQuery, [], dist, debug)

    if not cex:
        return [], initCondFailed

    cexs = [cex]
    initConditionFailed = initCondFailed
    i = 0
    from plotting_utils import plt
    # get symmetric points
    xlabel = vars[0]
    ylabel = vars[1]
    sym_pts = []
    for cex in cexs:
        pt = (cex[xlabel], cex[ylabel])
        symmPts = shape_utils.getSymmetricPts(shapeCoefs, pt)
        for p in symmPts:
            plt.plot(p[0], p[1], 'ro')
            if shape_utils.satisfiesPrecondition(p, program['inputRanges'], vars) \
                    or shape_utils.satisfiesInvariant(p, shapeCoefs, coefsToVars, outRanges, vars):
                pdict = {xlabel: p[0], ylabel: p[1]}
                sym_pts.append(pdict)
    cexs = cexs + sym_pts
    while i < numCex:
        i = i + 1
        # s, vars, mapVars, init, inductiveStep, cexs, dist
        new_cex, initCondFailed = checkInductiveWOCex(s, vars, mapVars, nondet, initQuery, indStepQuery, cexs, dist, outRanges, debug)
        if not new_cex or new_cex in cexs:  # can happen for symmetric points
            return cexs, initConditionFailed  # no new cex found, return what we have
        else:
            cexs.append(new_cex)
            initConditionFailed = initConditionFailed or initCondFailed

    return cexs, initConditionFailed


class TimeOutException(Exception):
    pass

