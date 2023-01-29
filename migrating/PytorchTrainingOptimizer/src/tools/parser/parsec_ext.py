from parsec import Parser, Value


def span(p: Parser):
    @Parser
    def span_parser(text, index=0):
        """ Parse until p parse failed
        """
        start = index
        result = p(text, index)
        while result.status:
            index = result.index
            result = p(text, index)
        return Value.success(index, text[start:index])
    return span_parser


def span_not(p: Parser):
    @Parser
    def span_not_parser(text, index=0):
        """ Parse until p parse success
        Caution: this parser step one step for each failure p, so it maybe contain some
        backtraces when p parse more than one charactor
        """
        start = index
        result = p(text, index)
        while not result.status:
            index += 1
            result = p(text, index)
        return Value.success(index, text[start:index])
    return span_not_parser


def not_empty(p: Parser):
    @Parser
    def not_empty_parser(text, index=0):
        """ Combine parser with a not-empty check
        """
        result = p(text, index)
        if not result.status:
            return result
        if result.value:
            return result
        return Value.failure(index, 'not empty string')
    return not_empty_parser


def escape(p: Parser):
    @Parser
    def escape_parser(text, index=0):
        """ Transform parser as an escape parser
        """
        if index >= len(text):
            return Value.failure(index, 'not eof')
        if text[index] != '\\':
            return Value.failure(index, 'escape mark [\\]')
        if index + 1 >= len(text):
            return Value.failure(index + 1, 'not eof after [\\]')
        result = p(text, index + 1)
        if not result.status:
            return result
        return Value.success(result.index, text[index:result.index])
    return escape_parser