# 18-05-2022

EOD: We got the model to init. 
For model forward we need inputs.

We're almost done getting inputs in their natural form (the way the appear). We need to convert them to tensors, 
ready to train and everything.

# 08-07-2022
In Dataiter and Loops
    - Dataiter is designed to work for all epochs. That is a bad idea, in my opinion. 
        We need to refactor it to start anew each epoch. Is that a bad idea? 
        Alternatively, we need to refactor the manner in which we're using the iterator.
        Does it give us the notion of an epoch? What happens when training stopped midway?
            - can we rerun the training then? etc etc

In Run, and Loops
    - Model saving

In Model
    - tie weights
    - embedding and other initialisation
