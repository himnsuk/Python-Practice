def num_cominations_for_final_score(final_score, individual_play_scores):
    # One way to reach 0.
    num_combinations_for_score = [[1] + [0] * final_score for _ in individual_play_scores]

    for i in range(len(individual_play_scores)):
        for j in range(1, final_score + 1):
            without_this_play = (num_combinations_for_score[i-1][j] if i >= 1 else 0)
            
            with_this_play = (num_combinations_for_score[i][j - individual_play_scores[i]]
            if j >= individual_play_scores[i] else 0)

            num_combinations_for_score[i][j] = (
                without_this_play + with_this_play)

    return num_combinations_for_score


final_score = 12
individual_play_scores = [2,3,7]

x = num_cominations_for_final_score(final_score, individual_play_scores)
print([i for i in range(1, final_score + 1)])
for ind, d in enumerate(x):
    print(f"{individual_play_scores[0: ind + 1]} -> {d}")

# Output when final_score = 12
#                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# [2]         -> [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
# [2, 3]      -> [1, 0, 1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 3]
# [2, 3, 7]   -> [1, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4]

# Output when final_score = 15
# [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
# [1, 0, 1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 3, 2, 3, 3]
# [1, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5]