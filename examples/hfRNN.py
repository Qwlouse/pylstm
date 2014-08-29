from pylstm import *


#global_rnd.set_seed(1234)

#BUILD LSTM NETWORK
#net = build_net(InputLayer(4) >> RnnLayer(100, act_func='tanh') >> HfFinalLayer(4, name='outputlayer', act_func='softmax'))
net = build_net(InputLayer(4) >> RnnLayer(50, act_func='tanh') >> RnnLayer(50, act_func='tanh') >> HfFinalLayer(4, name='outputlayer', act_func='softmax'))


#net = build_net(InputLayer(4) >> RnnLayer(100, act_func='tanh') >> ForwardLayer(4, name='outputlayer', act_func='softmax'))
net.error_func = MultiClassCrossEntropyError


net.initialize(default=Gaussian(std=.25), RnnLayer_1={'HR': SparseInputs(Gaussian(std=.25), connections=15),
                                                     'HX': SparseOutputs(Gaussian(std=1), connections=15),
                                                     },
                                          RnnLayer_2={'HR': SparseInputs(Gaussian(std=.25), connections=15),
                                                     'HX': SparseOutputs(Gaussian(std=1), connections=15),
                                                     },
               outputlayer={'HX': SparseInputs(Gaussian(std=.25), connections=15)})


#net.initialize(default=Gaussian(std=0.01))


def print_lambda(epoch, stepper, **_):
    print('lambda:', stepper.lambda_)


# MAKE 5 BIT PROBLEM
X, T = generate_memo_task(5,  2, 32, 50)
T = create_targets_object(T)
# Set up an SGD trainer that stops after 10 epochs
#tr = Trainer(net, SgdStep(learning_rate=.01))
#tr.stopping_criteria.append(MaxEpochsSeen(100))
#tr.monitor[''] = MonitorClassificationError(Online(X,T))
#tr.train(Undivided(X,T))

# Set up HF trainer 
tr = Trainer(net, CgStep(matching_loss=True))

# Train with weight updates after each sample
tr.stopping_criteria.append(MaxEpochsSeen(5000))
tr.add_monitor(MonitorClassificationError(Online(X,T)))
tr.add_monitor(PrintError())
tr.add_monitor(print_lambda)
tr.train(Undivided(X, T))
