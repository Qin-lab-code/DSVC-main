from SELFRec import SELFRec
from util.conf import ModelConf

if __name__ == '__main__':
    # Register your model here
    ssl_graph_models = ['DSVC']

    print('=' * 80)
    print('   SELFRec: A library for self-supervised recommendation.   ')
    print('=' * 80)
    print('Self-Supervised  Graph-Based Models:')
    print('   '.join(ssl_graph_models))
    print('=' * 80)
    model = input('Please enter the model you want to run:')
    import time

    s = time.time()
    if model in ssl_graph_models:
        conf = ModelConf('./conf/' + model + '.conf')
    else:
        print('Wrong model name!')
        exit(-1)
    rec = SELFRec(conf)
    rec.execute()
    e = time.time()
    print("Running time: %f s" % (e - s))
