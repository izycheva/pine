import ast
import math
import random
import re

# comment if you don't want determinism
# random.seed(7)


#  processing functions
def parse(inputFile):
    """ Parses the input program and returns it as a dictionary:
      { vars: ['x', 'y'], 
        inputRanges: {x: (a, b), y: (c, d)},
        loopBody: [[condition: string,...],{x: string, y: string}, ...],
        nondet: ['n0', 'n1'],
        assume: ['x <= n0*y']
      } """
    res = {'vars': [], 'inputRanges': {}, 'loopBody': {}, 'nondet': [], 'assume': []}

    f = open(inputFile, 'r')
    currentCondition = []
    reassignedVars = set()
    for line in f:
        # if-then-else condition
        if 'if' in line:
            cond_start = line.rstrip().lstrip().split('(')[1:]
            cond_ends = '('.join(cond_start).rstrip().lstrip().split(')')
            cond = ')'.join(cond_ends[:len(cond_ends) - 1])
            currentCondition.append(cond)
        elif 'else' in line:
            lastInd = len(currentCondition) - 1
            neg = 'not ' + str(currentCondition[lastInd])
            currentCondition = currentCondition[:lastInd]
            currentCondition.append(neg)
        # parse assumptions (need for handling bounded loops)
        elif 'assume:' in line:
            assume = line.split('assume:')[1].rstrip().lstrip()
            res['assume'].append(assume)
        # range spec
        elif '\\in' in line:
            split = line.rstrip().split('\\in')
            r = split[1].replace('[', '').replace(']', '').split(',')
            # res['vars'].append(split[0].rstrip())
            var = split[0].rstrip().lstrip()
            if var not in res['vars']:
                res['vars'].append(var)

            res['inputRanges'][split[0].rstrip().lstrip()] = (float(r[0].lstrip()), float(r[1].lstrip()))

        # loop body
        elif ':=' in line:
            split = line.rstrip().split(':=')
            var = split[0].rstrip().lstrip()
            if var not in res['vars']:
                res['vars'].append(var)
            reassignedVars.add(var)

            condition = ' and '.join(currentCondition) if currentCondition else 'true'
            if condition in res['loopBody']:
                res['loopBody'][condition][var] = split[1].lstrip().rstrip()
            else:
                res['loopBody'][condition] = {}
                res['loopBody'][condition][var] = split[1].lstrip().rstrip()

    # non-deterministic noise terms are the variables that have a range but not assigned a value in the loop
    nondetSet = set(res['vars']) - reassignedVars

    # to keep the order deterministic, use the order in the original list
    nondetOnly = [v for v in res['vars'] if v in nondetSet]
    varsOnly = [v for v in res['vars'] if v not in nondetSet]
    res['nondet'] = nondetOnly
    res['vars'] = varsOnly

    # fill identity function for each missing else branch
    # TODO handle nested conditions correctly
    thenBranches = filter((lambda x: 'not' not in x), list(res['loopBody'].keys()))
    for b in thenBranches:
        if b == 'true':
            continue
        elze = 'not ' + b
        if elze not in res['loopBody'].keys():
            res['loopBody'][elze] = {}
            for v in reassignedVars:
                root = 'true' in res['loopBody']
                # if the variable is not yet assigned one level higher
                res['loopBody'][elze][v] = v if not root or v not in res['loopBody']['true'] else \
                    res['loopBody']['true'][v]
    return res


def getCompiledLoopStatements(program):
    # generate and compile code to run simulation
    # 0.68*(x - y) -> 0.68*(inputs['x'] - inputs['y'])
    compiledLoopStmts = {}
    for branch in program['loopBody']:
        # negated cases handled below
        if 'not' in branch:
            continue
        # get all statements in the else-branch TODO adjust this part for nested if-then-else statements
        elseStmts = '' if branch == 'true' else program['loopBody'][f'not {branch}']
        for v, body in program['loopBody'][branch].items():
            tmp = body if branch == 'true' else f'{body} if {branch} else {elseStmts[v]}'
            sortedVars = sorted(program['vars'] + program['nondet'], key=len, reverse=True)
            for w in sortedVars:
                # match the variable name exactly
                tmp = re.sub(f'(?<![\[\'|a-z|A-Z]){w}(?![a-z|A-Z|\'\]])', f'inputs[\'{w}\']', tmp)

            # compiles code to be run directly in python
            compiledLoopStmts[v] = compile(tmp, '<string>', 'eval')
    return compiledLoopStmts


def simulate(program, m, n, benchName):
    """ Simulates the loop for n iterations for m random inputs"""
    # random.seed(7)
    # initialize points list: {'x': [], 'y': []}
    points = dict(map(lambda x: (x, []), program['vars']))

    # generate and compile code to run simulation
    # 0.68*(x - y) -> 0.68*(inputs['x'] - inputs['y'])
    compiledLoopStmts = getCompiledLoopStatements(program)

    i = 0
    while i < m:
        # draw inputs at random
        inputs = dict([(x, random.uniform(float(lo), float(up))) for x, (lo, up) in program['inputRanges'].items()])

        # add inputs to points
        for x in program['vars']:
            if x in inputs:
                points[x].append(inputs[x])
            # TODO the var is computed (no input range)
            else:
                print(str(x) + ' is not in input ranges')
                points[x] = []

        j = 0
        while j < n:
            z = {}
            # execute each loop body statement once and update points list
            for x, code in compiledLoopStmts.items():
                z[x] = eval(code)
                points[x].append(z[x])

            # update inputs list
            for x in program['vars']:
                inputs[x] = z[x]
            # vars that have input range but not assigned in the loop are non-deterministic inputs
            for x in program['nondet']:
                (lo, up) = program['inputRanges'][x]
                inputs[x] = random.uniform(float(lo), float(up))
            j = j + 1

        # end of outer while loop
        i = i + 1
    return points


def evalLoopOnce(loopStmts, progVars, pnts, nondet={}):
    """ computes the effect of one loop iteration for each point in points
    :param loopStmts: compiled loop statements
    :param vars: programVars
    :param pnts: list of points
    :param nondet: dict of non-deterministic vars and their ranges
    """
    new_points = []
    inputs = {}
    nondetvars = list(nondet.keys())
    for p in pnts:
        for i in range(0, len(progVars)):
            inputs[progVars[i]] = p[i]  # setup inputs for evaluation
        for i in range(len(nondetvars)):
            x = nondetvars[i]
            if len(p) > len(progVars) + i:
                inputs[x] = p[len(progVars) + i]
            else:
                lo, up = nondet[x]
                inputs[x] = random.uniform(float(lo), float(up))
        z = []
        # execute each loop body statement once and update points list
        for v in progVars:
            code = loopStmts[v]
            res = eval(code)
            if math.isinf(res):
                raise Exception(f'Sampling lead to +/- infinity for {v} on inputs {inputs}')
            if math.isnan(res):
                raise Exception(f'Sampling lead to NaN for {v} on inputs {inputs}')
            z.append(res)
        for x in nondetvars:
            z.append(inputs[x])

        new_points.append(z)
    return new_points


class ParenthesisVisitor(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.withParenthesis = ""

    def visit_BinOp(self, node):
        """ Enclose the operation in parentheses. """
        if isinstance(node.op, ast.Add):
            withParenthesis = '(' + self.visit(node.left) + ' + ' + self.visit(node.right) + ')'
        elif isinstance(node.op, ast.Mult):
            withParenthesis = '(' + self.visit(node.left) + ' * ' + self.visit(node.right) + ')'
        elif isinstance(node.op, ast.Sub):
            withParenthesis = '(' + self.visit(node.left) + ' - ' + self.visit(node.right) + ')'
        elif isinstance(node.op, ast.Div):
            withParenthesis = '(' + self.visit(node.left) + ' / ' + self.visit(node.right) + ')'
        self.withParenthesis = withParenthesis
        return withParenthesis

    def visit_UnaryOp(self, node):
        if isinstance(node.op, ast.USub):
            withParenthesis = '(-' + self.visit(node.operand) + ')'
        else:
            raise Exception("Unary operators except negation are not defined")
        self.withParenthesis = withParenthesis
        return withParenthesis

    def visit_Num(self, node):
        self.withParenthesis = f'{node.n}'
        return f'{node.n}'

    def visit_Name(self, node):
        self.withParenthesis = node.id
        return node.id


class ComparisonVisitor(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.exprs = []

    def visit_BoolOp(self, node):
        # boolean operations
        exprs = [self.visit(node.values[0])] + self.visit(node.op) + [self.visit(node.values[1])]
        self.exprs = exprs
        return exprs

    def visit_Compare(self, node):
        # consider only single comparison operations at the moment
        if isinstance(node.ops[0], ast.LtE):
            op = '<='
        elif isinstance(node.ops[0], ast.Lt):
            op = '<'
        elif isinstance(node.ops[0], ast.GtE):
            op = '>='
        elif isinstance(node.ops[0], ast.Gt):
            op = '>'
        elif isinstance(node.ops[0], ast.Eq):
            op = '=='
        exprs = [self.visit(node.left), op, self.visit(node.comparators[0])]
        self.exprs = exprs
        return exprs

    def visit_Num(self, node):
        self.exprs = [f'{node.n}']
        return f'{node.n}'

    def visit_Name(self, node):
        self.exprs = [node.id]
        return node.id

    def visit_BinOp(self, node):
        """ Enclose the operation in parentheses. """
        if isinstance(node.op, ast.Add):
            withParenthesis = '(' + self.visit(node.left) + ' + ' + self.visit(node.right) + ')'
        elif isinstance(node.op, ast.Mult):
            withParenthesis = '(' + self.visit(node.left) + ' * ' + self.visit(node.right) + ')'
        elif isinstance(node.op, ast.Sub):
            withParenthesis = '(' + self.visit(node.left) + ' - ' + self.visit(node.right) + ')'
        elif isinstance(node.op, ast.Div):
            withParenthesis = '(' + self.visit(node.left) + ' / ' + self.visit(node.right) + ')'
        self.exprs = [withParenthesis]
        return withParenthesis

    def visit_UnaryOp(self, node):
        """ Define a Real operation in SMT query for a unary minus and negation operation in the loop. """
        if isinstance(node.op, ast.USub):
            expr = f'-{self.visit(node.operand)}'
        elif isinstance(node.op, ast.Not):
            expr = f'not {self.visit(node.operand)}'
        else:
            raise Exception("Unary operators except negation are not defined")
        self.exprs = [expr]
        return expr