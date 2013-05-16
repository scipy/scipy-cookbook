TableOfContents(2)

Introduction
============

In this cookbook, we will focus on using
`pyparsing <http://pyparsing.wikispaces.com/>`__ and numpy to read a
structured text file like this one, `data.txt <.. image:: Reading_Custom_Text_Files_with_Pyparsing_attachments/data3.txt>`__:



.. code-block:: python

    # This is is an example file structured in section
    # with comments begining with '#'
    
    [ INFOS ]
    Debug          = False
    Shape  (mm^-1) = 2.3                                                            
    # here is a unit
    Length (mm)    = 25361.15
    Path 1         = C:\\This\is\a\long\path\with some space in it\data.txt
    description    = raw values can have multiple lines, but additional lines must s
    tart
                     with a whitespace which is automatically skipped
    Parent         = None
    
    [ EMPTY SECTION ]
    # empty section should not be taken into account
    
    [ TABLE IN ROWS ]
    Temp    (C)             100             200        300       450.0        600
    E XX    (GPa)           159.4       16.9E+0       51.8      .15E02        4     
    # Here is a space in the row name
    Words               'hundred'   'two hundreds'  'a lot'     'four'      'five'  
    # Here are QuotedStrings with space
    
    [ TABLE IN COLUMNS ] 
    STATION         PRECIPITATION   T_MAX_ABS  T_MIN_ABS 
    (/)                     (mm)    (C)        (C)       # Columns must have a unit
    Ajaccio                 64.8    18.8E+0    -2.6      
    Auxerre                 49.6    16.9E+0    Nan       # Here is a Nan
    Bastia                  114.2   20.8E+0    -0.9      
    
    [ MATRIX ]
    True    2       3
    4.      5.      6.
    7.      nan     8
    



and we will create a reusable parser class to automatically:

| ``* detect section blocs, among four possible kinds :``
| `` * a set of variable declarations : ``\ *``name``*\ `` (``\ *``unit``*\ ``) = ``\ *``value``*\ ``, ``\ *``unit``*\ `` is optional``
| `` * a table defined row by row, where the first column defines the name of the row. This name can have spaces in it if it is followed by an unit, otherwise it can't.``
| `` * a table defined column by column. Column names can't contain spaces and the second row should in this case contains units``
| `` * a matrix containing only numeric values, True, False or NaN``
| ``* convert values into the adequate Python or Numpy type (True, False, None, NaN, float, str or array)``
| ``* detect associated units if present``
| ``* return a data structure with the same organization in section as the input file and clean up variable name to get a name compatible with named attribute access``

Here is a session example with this parser,
`ConfigNumParser <.. image:: Reading_Custom_Text_Files_with_Pyparsing_attachments/ConfigNumParser_v0.1.1.py>`__:



.. code-block:: python

    >>> from ConfigNumParser import *
    >>> data = parseConfigFile('data.txt')
    >>> pprint(data.asList())
    [['infos',
      ['debug', False],
      ['shape', 2.2999999999999998],
      ['length', 25361.150000000001],
      ['path_1', 'C:\\\\This\\is\\a\\long\\path\\with some space in it\\data.txt'],
      ['description',
       'raw values can have multiple lines, but additional lines must start\nwith a 
    whitespace which is automatically skipped'],
      ['parent', None],
      ['names_', ['debug', 'shape', 'length', 'path_1', 'description', 'parent']],
      ['unit_', {'length': 'mm', 'shape': 'mm^-1'}]],
     ['table_in_rows',
      ['temp', array([ 100.,  200.,  300.,  450.,  600.])],
      ['e_xx', array([ 159.4,   16.9,   51.8,   15. ,    4. ])],
      ['words', array(['hundred', 'two hundreds', 'a lot', 'four', 'five'], dtype='|
    S12')],
      ['names_', ['temp', 'e_xx', 'words']],
      ['unit_', {'e_xx': 'GPa', 'temp': 'C'}]],
     ['table_in_columns',
      ['station', array(['Ajaccio', 'Auxerre', 'Bastia'], dtype='|S7')],
      ['precipitation', array([  64.8,   49.6,  114.2])],
      ['t_max_abs', array([ 18.8,  16.9,  20.8])],
      ['t_min_abs', array([-2.6,  NaN, -0.9])],
      ['names_', ['station', 'precipitation', 't_max_abs', 't_min_abs']],
      ['unit_',  {'precipitation': 'mm', 't_max_abs': 'C', 't_min_abs': 'C'}]],
     ['matrix',
      array([[  1.,   2.,   3.],
           [  4.,   5.,   6.],
           [  7.,  NaN,   8.]])]]
    
    >>> data.matrix
    array([[  1.,   2.,   3.],
           [  4.,   5.,   6.],
           [  7.,  NaN,   8.]])
    
    >>> data.table_in_columns.t_max_abs
    array([ 18.8,  16.9,  20.8])
    
    >>> data.infos.length, data.infos.unit_['length']
    (25361.15, 'mm')
    



This parser add two specials fields in all sections but matrix ones :

| ``* ``\ *``names_``*\ `` : a list containing the names of all variables found in this section``
| ``* ``\ *``unit_``*\ `` : a dict containing the unit corresponding to each variable name, if there is any``

Defining a parser for parameter declarations
============================================

`pyparsing <http://pyparsing.wikispaces.com/>`__ is an efficient tool to
deal with formatted text, and let you process in two steps:

``1. Define rules to identify strings representing sections, variable names, and so on.  With pyparsing, theses rules can be combined easily with the standard operators | and + and creating reusable components becomes  easy too. ``

``1. Define actions to be executed on theses fields, to convert them into python objects.``

In the file example above, there are four kinds of data: parameter
definitions, table in rows, table in columns and matrix.

So, we will define a parser for each one and combine them to define the
final parser.

First steps with pyparsing
--------------------------

This section will describe step by step how to build the function
\`paramParser\` defined in
`ConfigNumParser <.. image:: Reading_Custom_Text_Files_with_Pyparsing_attachments/ConfigNumParser_v0.1.1.py>`__, used to
parse the bloc [ INFOS ] in the example above.

A parameter declaration has the form:

| ``   ``\ *``key``*\ `` (``\ *``unit``*\ ``) = ``\ *``value``*
| ``     ``
| `` with:``

| ``    * ``\ *``key``*\ ``  : a set of alphanumeric characters or _``
| ``    * ``\ *``unit``*\ `` : an optional set of alphanumeric characters or ^ * / - . _``
| ``    * ``\ *``value``*\ ``: anything to the end of line or to the character # which starts a comment``

This can be translated almost literally with pyparsing syntax (see `how
to use pyparsing <http://pyparsing.wikispaces.com/HowToUsePyparsing>`__
for more information):



.. code-block:: python

    from    pyparsing   import *
    # parameter definition
    keyName       = Word(alphanums + '_')
    unitDef       = '(' + Word(alphanums + '^*/-._') + ')'
    paramValueDef = SkipTo('#'|lineEnd)
    
    paramDef = keyName + Optional(unitDef) + "=" + empty + paramValueDef
    



It is easy to test what will be found with this pattern in the data
file:



.. code-block:: python

    # print all params found
    >>> for param in paramDef.searchString(file('data.txt').read()):
    ...     print param.dump()
    ...     print '...'
    ['Context', '=', 'full']
    ...
    ['Temp_ref', '(', 'K', ')', '=', '298.15']
    ...
    ...
    



We can improved it in a few ways:

| ``* suppress meaningless fields '(', '=', ')' from the output, with the use of the `Suppress` element, ``
| ``* give a name to the different fields, with the `setResultsName` method, or simply just by calling an element with the name in argument``



.. code-block:: python

    # parameter definition
    keyName       = Word(alphanums + '_')
    unitDef       = Suppress('(') + Word(alphanums + '^*/-._') + Suppress(')')
    paramValueDef = SkipTo('#'|lineEnd)
    
    paramDef = keyName('name') + Optional(unitDef)('unit') + Suppress("="+empty) + p
    aramValueDef('value')
    



The test will now give name to results and gives a nicer output:



.. code-block:: python

    ['Context', 'full']
    - name: Context
    - value: full
    ...
    ['Temp_ref', 'K', '298.15']
    - name: Temp_ref
    - unit: ['K']
    - value: 298.15
    ...
    ...
    



Converting data into Python objects
-----------------------------------

We will detail further what kind of values are expected to let pyparsing
handle the conversion.

They can be divided in two parts :

| ``* Python objects like numbers, True, False, None, NaN or any string between quotes.``
| ``* Raw strings that should not be converted``

Let's begin with numbers. We can use the \`Regex\` element to rapidly
detect strings representing numbers:



.. code-block:: python

    from re        import VERBOSE
    number = Regex(r"""
            [+-]?                           # optional sign
             (
                (?:\d+(?P<float1>\.\d*)?)   # match 2 or 2.02
              |                             # or
                (?P<float2>\.\d+)           # match .02
             )
             (?P<float3>[Ee][+-]?\d+)?      # optional exponent
            """, flags=VERBOSE
            )
    



See `Regular expression
operations <http://docs.python.org/library/re.html#module-re>`__ for
more information on regular expressions. We could have built a parser
with standard pyparsing elements (\`Combine\`, \`Optional\`, \`oneOf\`,
etc.) but low-level expressions like floating point numbers are said to
do really much better using the \`Regex\` class. I know it feels like
cheating, but in truth, pyparsing uses a number of re's under the
covers.

Now we will define a function to convert this string into python float
or integer and set a \`parseAction\` to tell pyparsing to automatically
convert a number when it find one:



.. code-block:: python

    def convertNumber(t):
        """Convert a string matching a number to a python number"""
        if t.float1 or t.float2 or t.float3 : return [float(t[0])]
        else                                : return [int(t[0])  ]
    
    number.setParseAction(convertNumber)
    



The \`convertNumber\` function is a simple example of \`parseAction\`:

``* it should accepts a `parseResults` object as input value (some functions can accepts 3 parameters, see `setParseAction` documentation). A `parseResults` object can be used as a list, as a dict or directly with a named attribute if you have named your results. Here we had set three named group float1, float2 and float3 and we can use them to decide whether to use int() or float().``

``* it should return either a `parseResults` object or a list of results which will be automatically converted to a `parseResults` object.``

Pyparsing comes with a very convenient function to convert fields to a
constant object, namely \`replaceWith\`. This can be used to create a
list of element converting strings to python objects:



.. code-block:: python

    from numpy     import NAN
    
    pyValue_list = [ number                                                        ,
    
                     Keyword('True').setParseAction(replaceWith(True))             ,
    
                     Keyword('False').setParseAction(replaceWith(False))           ,
    
                     Keyword('NAN', caseless=True).setParseAction(replaceWith(NAN)),
    
                     Keyword('None').setParseAction(replaceWith(None))             ,
    
                     QuotedString('"""', multiline=True)                           ,
    
                     QuotedString("'''", multiline=True)                           ,
    
                     QuotedString('"')                                             ,
    
                     QuotedString("'")                                             ,
    
                   ]
    
    pyValue     = MatchFirst( e.setWhitespaceChars(' \t\r') for e in pyValue_list)
    



Here we used:

| ``* `Keyword` to detect standard python keyword and replace them on the fly  ``
| ``* `QuotedString` to detect quoted string and automatically unquote them``
| ``* `MatchFirst` to build a super element, `pyValue` to convert all kind of python values.``

Let's see what we get:



.. code-block:: python

    >>> test2 = '''
    >>>     1   2   3.0  0.3 .3  2e2  -.2e+2 +2.2256E-2
    >>>     True False nan NAN None
    >>>     "word" "two words"
    >>>     """'more words', he said"""
    >>> '''
    >>> print pyValue.searchString(test2)
    [[1], [2], [3.0], [0.29999999999999999], [0.29999999999999999], [200.0], [-20.0]
    , [0.022256000000000001],
    [True], [False], [nan], [nan], [None], ['word'], ['two words'], ["'more words', 
    he said"]]
    



Some words on whitespace characters
-----------------------------------

By default, pyparsing considers any characters in ' \\t\\r\\n') as
whitespace and meaningless. If you need to detect ends-of-line you need
to change this behavior by using \`setWhitespaceChars\` or
\`setDefaultWhitespaceChars\`.

As we are going to process tables line by line, we need to configure
this and this should be set up *at the lowest level*:



.. code-block:: python

    >>> pyValue2     = MatchFirst(pyValue_list)          # default behavior
    >>> print OneOrMore(pyValue2).searchString(test2)
    [[1, 2, 3.0, 0.29999999999999999, 0.29999999999999999, 200.0, -20.0, 0.022256000
    000000001, True, False, nan, nan, None, 'word', 'two words', "'more words', he s
    aid"]]
    
    >>> # to compare to
    
    >>> for r, s, t in OneOrMore(pyValue).searchString(test2)
    [[1, 2, 3.0, 0.29999999999999999, 0.29999999999999999, 200.0, -20.0, 0.022256000
    000000001],
    [True, False, nan, nan, None],
    ['word', 'two words'],
    ["'more words', he said"]]
    



Converting variables names
--------------------------

We must also detail what is an acceptable parameter name.

As the end of the parameter name is delimited by the = character, we
could accept to have spaces in it. But as we want the possibility to
access to its value via a named attribute, we need to convert it to a
standard form, compatible with python's naming conventions. Here we
choose to format parameter names to lowercase, with any set of character
in ' -/.' replaced with underscores.

Later, we will have to deal with parameter names where spaces can't be
allowed. So we will have to define two kind of names:



.. code-block:: python

    def variableParser(escapedChars, baseChars=alphanums):
        """ Return pattern matching any characters in baseChars separated by
        characters defined in escapedChars. Thoses characters are replaced with '_'
    
        The '_' character is therefore automatically in escapedChars.
        """
        escapeDef = Word(escapedChars + '_').setParseAction(replaceWith('_'))
        whitespaceChars = ''.join( x for x in ' \t\r' if not x in escapedChars )
        escapeDef = escapeDef.setWhitespaceChars(whitespaceChars)
        return Combine(Word(baseChars) + Optional(OneOrMore(escapeDef + Word(baseCha
    rs))))
    
    keyName             = variableParser(' _-./').setParseAction(downcaseTokens)
    keyNameWithoutSpace = variableParser('_-./').setParseAction(downcaseTokens)
    



\`downcaseTokens\` is a special pyparsing function returning every
matching tokens lowercase.

Dealing with raw text
---------------------

To finish this parser, we now need to add a rule to match raw text
following the conditions:

| ``* anything after the # character is considered as a comment and skipped``
| ``* a raw value can be on several lines, but the additional lines must start with a whitespace and not with a [``



.. code-block:: python

    # rawValue can be multiline but theses lines should start with a Whitespace
    rawLine  = CharsNotIn('#\n') + (lineEnd | Suppress('#'+restOfLine))
    rawValue = Combine( rawLine + ZeroOrMore(White(' \t').suppress()+ NotAny('[') + 
    rawLine))
    rawValue.setParseAction(lambda t: [x.strip() for x in t])
    



We will also refine our definition of units to handle special cases like
(-), (/) or (), corresponding to a blank unit.

This leads to:



.. code-block:: python

    unitDef  = Suppress('(') + (Suppress(oneOf('- /')) | Optional(Word(alphanums + '
    ^*/-._'))) + Suppress(')')
    valueDef = pyValue | rawValue
    paramDef = keyName('name') + Optional(unitDef)('unit') + Suppress("="+empty) + v
    alueDef('value')
    



Structuring data
----------------

We will try to organize the results in an easy to use data structure.

To do so, we will use the \`Dict\` element, which allows access by key
as a normal dict or by named attributes. This element takes for every
tokens found, its first field as the key name and the following ones as
values. This is very handy when you can group data with the \`Group\`
element to have only two fields.

As we can have three of them (with units) we will put these units aside:



.. code-block:: python

    def formatBloc(t):
        """ Format the result to have a list of (key, values) easily usable with Dic
    t
    
        Add two fields :
            names_ : the list of column names found
            units_ : a dict in the form {key : unit}
        """
        rows = []
    
        # store units and names
        units = {}
        names = []
    
        for row in t :
            rows.append(ParseResults([ row.name, row.value ]))
            names.append(row.name)
            if row.unit : units[row.name] = row.unit[0]
    
        rows.append( ParseResults([ 'names_', names ]))
        rows.append( ParseResults([ 'unit_',  units]))
    
        return rows
    
    paramParser = Dict( OneOrMore( Group(paramDef)).setParseAction(formatBloc))
    



This \`paramParser\` element is exactly the parser created by the
function \`paramParser\` defined in
`ConfigNumParser <.. image:: Reading_Custom_Text_Files_with_Pyparsing_attachments/ConfigNumParser_v0.1.1.py>`__.

Let's see what we get:



.. code-block:: python

    >>> paramParser.ignore('#' + restOfLine)
    >>> data = paramParser.searchString(file('data.txt').read())[0]
    >>> print data.dump()
    [...]
    - debug: False
    - description: raw values can have multiple lines, but additional lines must sta
    rt
    with a whitespace which is automatically skipped
    - length: 25361.15
    - names_: ['debug', 'shape', 'length', 'path_1', 'description', 'parent']
    - parent: None
    - path_1: 'C:\\This\is\a\long\path\with some space in it\data.txt'
    - shape: 2.3
    - unit_: {'shape': 'mm^-1', 'length': 'mm'}
    >>> data.length, data.unit_['length']
    Out[12]: (25361.150000000001, 'mm')
    



Defining a parser for tables
============================

For parsing parameter declarations, we have seen most of the common
techniques but one: the use of \`Forward\` element to define parsing
rules on the fly.

Let's see how this can be used to parse a table defined column by
column, according to this schema:



.. code-block:: python

                Name_1       Name_2     ...      Name_n
                (unit_1)    (unit_2)    ...     (unit_n)
                value_11    value_21    ...     value_n1
                  ...         ...       ...       ...
    



and the following rules:

| ``* Names can't contains any whitespaces.``
| ``* Units are mandatory.``
| ``* Value can be any standard python value (int, number, None, False, True, NaN or quoted strings) or a raw string which can't contains spaces or '['.``

Such a parser can be generated with the \`tableColParser\` function
defined in `ConfigNumParser <.. image:: Reading_Custom_Text_Files_with_Pyparsing_attachments/ConfigNumParser_v0.1.1.py>`__.

The heart of the problem is to tell pyparsing that each line should have
the same number of columns, whereas this number is unknown a priori.

Using the Forward element
-------------------------

We will get round this problem by defining the pattern corresponding to
the unit line and its followers right after reading the header line.

Indeed, theses lines can be defined with a \`Forward\` element and we
can attach a \`parseAction\` to the header line to redefine these
elements later, once we know how many columns we have in the headers.

Redefining a \`Forward\` element is done via the << operator:



.. code-block:: python

    # We define ends-of-line and what kind of values we expect in tables
    EOL          = LineEnd().suppress()
    tabValueDef  = pyValue | CharsNotIn('[ \t\r\n').setWhitespaceChars(" \t")
    
    # We define how to detect the first line, which is a header line
    # following lines will be defined later
    firstLine    = Group(OneOrMore(keyNameWithoutSpace)+EOL)
    unitLine     = Forward()
    tabValueLine = Forward()
    
    def defineColNumber(t):
        """ Define unitLine and tabValueLine to match the same number of columns tha
    n
        the header line"""
        nbcols = len(t.header)
        unitLine      << Group( unitDef*nbcols + EOL)
        tabValueLine  << Group( tabValueDef*nbcols + EOL)
    
    tableColDef = (   firstLine('header').setParseAction(defineColNumber)
                    + unitLine('unit')
                    + Group(OneOrMore(tabValueLine))('data')
                  )
    



Structuring our data
--------------------

Now will organize our data the same way we did for parameters, but we
will use this time the name of the column as the key and we will
transform our data into numpy arrays:



.. code-block:: python

    def formatBloc(t):
        """ Format the result to have a list of (key, values) easily usable
        with Dict and transform data into array
    
        Add two fields :
            names_ : the list of column names found
            units_ : a dict in the form {key : unit}
        """
        columns = []
    
        # store names and units names
        names = t.header
        units   = {}
    
        transposedData = zip(*t.data)
        for header, unit, data in zip(t.header, t.unit, transposedData):
            units[header] = unit
            columns.append(ParseResults([header, array(data)]))
    
        columns.append(ParseResults(['names_', names]))
        columns.append(ParseResults(['unit_'   , units  ]))
    
        return columns
    
    tableColParser = Dict(tableColDef.setParseAction(formatBloc))
    



Let's see what we get:



.. code-block:: python

    >>> tableColParser.ignore('#' + restOfLine)
    >>> data = tableColParser.searchString(file('data3.txt').read())[0]
    >>> print data.dump()
    [...]
    - names_: ['station', 'precipitation', 't_max_abs', 't_min_abs']
    - precipitation: [  64.8   49.6  114.2]
    - station: ['Ajaccio' 'Auxerre' 'Bastia']
    - t_max_abs: [ 18.8  16.9  20.8]
    - t_min_abs: [-2.6  NaN -0.9]
    - unit_: {'station': '/', 'precipitation': 'mm', 't_min_abs': 'C', 't_max_abs': 
    'C'}
    



Building the final parser
=========================

We have now three kinds of parsers:

| ``* `variableParser  :` handle variables names``
| ``* `paramParser     :` handle a set of variable definitions``
| ``* `tableColParser  :` handle tables defined column by column``

There are two more in
`ConfigNumParser <.. image:: Reading_Custom_Text_Files_with_Pyparsing_attachments/ConfigNumParser_v0.1.1.py>`__ :

| ``* `tableRowParser  :` handle tables defined row by row``
| ``* `matrixParser    :` handle matrix containg only python values or NaN``

We won't detail them here, because they use exactly the same techniques
we've already seen.

We will rather see how to combine them into a complex parser as it is
done in the \`parseConfigFile\` function:



.. code-block:: python

    # Section header
    sectionName = Suppress('[') + keyName + Suppress(']')
                                                                  
    # Group section name and content 
    section = Group (sectionName +
                      ( paramParser()
                      | tableColParser()
                      | tableRowParser()
                      | matrixParser()
                )     )
                                                                  
    # Build the final parser and suppress empty sections
    parser = Dict( OneOrMore( section | Suppress(sectionName) )) 
                                                                  
    # Defines comments
    parser.ignore('#' + restOfLine)
    



That's all.

The parser can now be use through its method \`parseString\` or
\`parseFile\`. See
`ConfigNumParser <.. image:: Reading_Custom_Text_Files_with_Pyparsing_attachments/ConfigNumParser_v0.1.1.py>`__ for more
information.

I hope this will give you a good starting point to read complex
formatted text.

