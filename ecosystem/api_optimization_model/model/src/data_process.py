import knowledges

def data_process(file_pathname, extend_result):
    # 遍历该目录下的所有code文件
    dvppmem = []
    if not os.path.isdir(file_pathname):
        return extend_result

    for filename in os.listdir(file_pathname):
        if filename.endswith('.cpp') or filename.endswith('.py') or filename.endswith('.h'):
            line_num = 0
            path = os.path.join(file_pathname, filename)
            with open(path, encoding='UTF-8') as f:
                contents = f.readlines()
                for line in contents:
                    line_num += 1
                    # 0 copy
                    for k, r in knowledges.knowledges #遍历知识库
                        if line in k :
                            value = []
                            value.append(k)
                            value.append(r)
                            value.append(filename + ' Line:' + str(line_num))
                            extend_result.value.append(value)

    return extend_result