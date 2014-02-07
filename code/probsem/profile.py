from learn import Learner
from learn_test import train



def run():
    train_data = train()
    data = [train_data[0], train_data[5]]
    learner = Learner()
    learner.hidden_dims = 2
    learner.dims['noun'] = 3
    learner.dims['det'] = 4
    learner.dims['np'] = 5
    learner.max_its = 1
    learner.learn(data)


if __name__ == "__main__":
    import cProfile
    import pstats

    cProfile.run('run()', 'profile_stats')
    p = pstats.Stats('profile_stats')
    p.sort_stats('cumulative').print_stats(50)
