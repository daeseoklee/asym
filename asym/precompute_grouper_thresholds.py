from typing import List, Callable
import random

"""
This module is suppose to be run indepently by users before using Grouper in their main application.      

Suppose the computational cost(time, memory etc.) of proccessing a batch depends only on its length. 
Let cost_fn be the approximation of the function (length -> cost). 
Then, a good set of thresholds [thres1, ..., thresk] of LengthThresholdGrouper for the dataset would satisfy the following condition: 
    cost_fn(thres1) * probability(len(data)<=thres1)  
    == cost_fn(thres2) * probability(thres1<len(data)<=thres2) 
    == cost_fn(thres3) * probability(thres2<len(data)<=thres3)
    .
    .
This module tries to find an approximate solution of it by using a greedy strategy. 
In the strategy, for a given multiset of lengths [l], 
    1. We randomly initialize the threshold values.
    2. We iteratively update them until we either 1)Find out that we have fallen into a repetition of period 2 or 2)We have tolerated symptoms of dead-ends [tolerate] times. 
The updates are done by:  
    1. Partition [l] according to the thresholds. (The partition would form "groups") 
    2. Compare for each threshold value the left-hand-side group and the right-hand-side group in their cost.(Cost that they would have after forming a batch) 
    3. Find the threshold value at which the difference is the largest, and either increment or decrement it by 1 accordingly. 

[_find_thresholds] implement the strategy described in the previous paragraph, 
while [find_thresholds] tries many episodes of it and report the best result.  
"""


def _find_thresholds(l:List[int], cost_fn:Callable[[int], int], k:int, tolerate:int=None):
    """
    Args:
        l: The lengths 
        cost_fn: The batch cost function 
        k: The number of thresholds
        tolerate: Described in the beginning

    Returns:
        (thresholds, max_diff, status) where
        -thresholds: the determined thresholds 
        -max_diff: The maximum difference of consecutive length groups as desdribed in the beginning. 
        -status: The reason that the search has stopped.  
    """
    l = sorted(l)
    min_x = l[0]
    max_x = l[-1]
    xs = [] #length values in the increasing order
    cum = {} #cumulative 
    cum[0] = 0 
    last_x = 0 
    for i, x in enumerate(l):
        if x <= 0:
            raise Exception('lengths must be positive integers')
        if x == last_x:
            cum[last_x] += 1
        else:
            assert x > last_x
            xs.append(x)
            cum[x] = cum[last_x] + 1
        last_x = x 
    if len(xs) < k + 1:
        raise Exception(f'{k} is too big')
    if tolerate is None:
        tolerate = 2 * len(xs)
    t_list = random.sample(list(range(len(xs))), k)
    t_list.sort()
    t = tuple(t_list) 
    #t: tuple of threshold indices of xs
    
    def get_costs(t):
        costs = []
        last_x = 0
        for j in range(k):
            x = xs[t[j]] #thresholding length
            cost = cost_fn(x) * (cum[x] - cum[last_x])
            costs.append(cost) 
            last_x = x 
        x = xs[-1] 
        cost = cost_fn(x) * (cum[x] - cum[last_x])
        costs.append(cost)
        return costs 
    
    def find_update(t):
        costs = get_costs(t) #len(costs) == k + 1
        max_diff = -1
        argmax_j = None
        update_direction = None
    
        for j in range(k):
            diff = costs[j + 1] - costs[j]
            if abs(diff) > max_diff:
                argmax_j = j 
                max_diff = abs(diff)
                if diff > 0:
                    update_direction = 1
                elif diff < 0:
                    update_direction = -1
                else:
                    update_direction = 0
        
        return (max_diff, argmax_j, update_direction) 
    
    def updated(t, j, direction):
        return t[:j] + (t[j] + direction,) + t[j+1:]
    
    prev_ts = [None, None]
    prev_max_diff = None
    tolerated = 0 
    status = None
    while True:
        max_diff, argmax_j, update_direction = find_update(t) 
        if update_direction == 0:
            status = 'perfect' 
            break 
        if prev_max_diff is not None and max_diff >= prev_max_diff:
            tolerated += 1
            if tolerated > tolerate:
                status = 'could not tolerate'
                t = prev_ts[1]
                max_diff = prev_max_diff
                break
            if prev_ts[0] == t:
                status = 'repeated'
                t = prev_ts[1]
                max_diff = prev_max_diff
                break 
        if update_direction == 1:
            if argmax_j == k - 1:
                if t[argmax_j] == len(xs) - 1:
                    status = 'extreme'
                    break 
            else:
                if t[argmax_j + 1] - t[argmax_j] == 1:
                    status = 'narrow gap'
                    break 
        else: 
            assert update_direction == -1
            if argmax_j == 0:
                if t[argmax_j] == 0:
                    status = 'extreme'
                    break
            else:
                if t[argmax_j] - t[argmax_j - 1] == 1:
                    status = 'narrow gap'
                    break 
        prev_max_diff = max_diff 
        prev_ts[0] = prev_ts[1] 
        prev_ts[1] = t
        t = updated(t, argmax_j, update_direction)
    thresholds = [xs[i] for i in t]
    return thresholds, max_diff, status

def find_thresholds(l, cost_fn, k, tolerate=None, print_status=False, num_trials=100, return_max_diff=False):
    """
    Args:
        l: The lengths 
        cost_fn: The batch cost function 
        k: The number of thresholds
        tolerate: Described in the beginning
        print_status(Optional): prints the stopping status frequency dict. 
        num_trials[Optional]: Number of trials of [_find_thresholds]
        return_max_diff[Optional]: .

    Returns:
        (thresholds, max_diff, status) where
        -thresholds: the determined thresholds 
        -max_diff: The maximum difference of consecutive length groups as desdribed in the beginning. 
        -status: The reason that the search has stopped.  
    """
    optimal_thresholds = None
    min_max_diff = None
    status_freq = {}
    for _ in range(num_trials):
        thresholds, max_diff, status = _find_thresholds(l, cost_fn, k, tolerate=tolerate)
        if min_max_diff is None or max_diff < min_max_diff:
            optimal_thresholds = thresholds
            min_max_diff = max_diff
        if status in status_freq:
            status_freq[status] += 1
        else:
            status_freq[status] = 1
    if print_status:
        print(status_freq) 
    if return_max_diff:
        return optimal_thresholds, min_max_diff
    return optimal_thresholds

def test_find_thresholds():
    #similar to 64 * 2 ^ 2 == 16 * 4 ^ 2 == 4 * 8 ^ 2 == 1 * 16 ^ 2
    n = 6
    l = [] 
    for i in range(1, n + 1):
        max_x = 2 ** i 
        min_x = 2 ** (i - 1) + 1
        l += [max_x]
        l += [random.randint(min_x, max_x) for _ in range(2 ** (2 * n - 2 * i) - 1)]
    thresholds, max_diff = find_thresholds(l, lambda x: x ** 2, n - 1, print_status=True, num_trials=1000, return_max_diff=True)
    print(thresholds)
    print(max_diff)

if __name__ == '__main__':
    test_find_thresholds()
    