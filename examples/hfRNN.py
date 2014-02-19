from pylstm import *

#BUILD LSTM NETWORK
net = build_net(InputLayer(4) >> RnnLayer(100, act_func='tanh') >> HfFinalLayer(4, act_func='softmax'))
net.error_func = MultiClassCrossEntropyError
net.initialize(default=Gaussian(std=.25), RnnLayer={'HR': SparseInputs(Gaussian(std=.25), connections=15),
                                                     'HX': SparseOutputs(Gaussian(std=1), connections=15),
                                                     },
               HfFinalLayer={'HX': SparseInputs(Gaussian(std=.25), connections=15)})


def print_lambda(epoch, stepper, **_):
    print('lambda:', stepper.lambda_)


#net.initialize(default=Gaussian(std=0.01))

# MAKE 5 BIT PROBLEM
X, T = generate_memo_task(5,  2, 32, 50)

# Set up an SGD trainer that stops after 10 epochs
#tr = Trainer(net, SgdStep(learning_rate=.01))
#tr.stopping_criteria.append(MaxEpochsSeen(100))
#tr.monitor[''] = MonitorClassificationError(Online(X,T))
#tr.train(Undivided(X,T))

# Set up HF trainer 
tr = Trainer(net, CgStep())

# Train with weight updates after each sample
tr.stopping_criteria.append(MaxEpochsSeen(5000))
tr.monitor[''] = MonitorClassificationError(Online(X,T))
tr.monitor['err'] = print_error_per_epoch
tr.monitor['lambda'] = print_lambda
tr.train(Undivided(X, T))
