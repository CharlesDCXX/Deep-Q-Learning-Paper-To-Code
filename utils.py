# 从键盘输入5个考试成绩，计算最高分，最低分和不及格的人数。

# 从键盘输入5个成绩，存储到列表中
scores = list(map(int, input().split()))

# 计算最高分、最低分、不及格人数
max_score = max(scores)
min_score = min(scores)
fail_num = len([score for score in scores if score < 60])

# 输出结果
print(max_score)
print(min_score)
print(fail_num)
