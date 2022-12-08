from os import remove
from parsec import ParseError, many, none_of, parsecmap, string, any
from enum import Enum
from functional import const, left_append, enclose_with
from parsec_ext import span, span_not, escape, not_empty


""" Parser for literal strings in python source file
There're two kinds of literal strings in python source file
1. enclosed with single quotation marks, e.g. 'abc""'
2. enclosed with double quotation marks, e.g. "abc''"
Escape charactor check is involved in both of kinds
"""
literal_p = (parsecmap(string("'") >> span(escape(any()) ^ none_of("'")) << string("'"), enclose_with("'"))) \
          | (parsecmap(string('"') >> span(escape(any()) ^ none_of('"')) << string('"'), enclose_with('"')))
""" Parser for inline comments
"""
inline_comment_p = parsecmap(string('#') >> span(none_of('\n')) < string('\n'), left_append('# '))
""" Parser for block comments
"""
block_comment_p = parsecmap(string('"""') >> span_not(string('"""')) << string('"""'), enclose_with('"""'))
""" Parser for remain codes
"""
code_p = not_empty(span(none_of('\'"#')))


def remove_comments(text):
    """ Remove inline and block comments in python source file
    :param text: python source
    """
    # Replace parse value with empty string ''
    remove_inline_p = parsecmap(inline_comment_p, const(''))
    remove_block_p  = parsecmap(block_comment_p, lambda s: '\n' * s.count('\n'))
    parser = many(remove_block_p ^ literal_p | code_p | remove_inline_p)
    try:
        return ''.join(parser.parse(text))
    except ParseError as e:
        print(e)
        return None