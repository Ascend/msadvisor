import pathlib


def set_up_helper(postfix, content, ext='py'):
    dir_ = pathlib.Path(f'../data/project/test_task_{postfix}/')
    dir_.mkdir(parents=True, exist_ok=True)
    file = pathlib.Path(f'test_{postfix}.{ext}')
    with open(pathlib.Path(dir_, file), 'w', encoding='utf-8') as fp:
        fp.write(content)
