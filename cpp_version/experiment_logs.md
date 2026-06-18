2026-04-16 v0:

The number of predictions oscilliate between 490 and 510 in multiple tries.
This seems indicating the manifold holds some properties of living organism.
Need further investigations on its behaviors.

Currently the output prediction accuracies control the scale of inputs in next round.

After increasing stimulation number from 10 to 100, the same resulting oscilliation. 490 and 510, shows up.

continus flip between 1 and 0 means negative signals.

Updated the F1-score calculation, now the score converging to 0.001 for some unknown reason.

2026-04-17 v0:

Change the initial counter strength from 1 to 0. To decrease noises generated inside manifolds, even when there is no input signals.

For some unknown reason, the flips between 0 and 1 keep showing up in output nodes.
Need to investigate the behavior property of such manifold connections; see how those properties can be leveraged to build a correct feedback mechanism.

Looks lke all connections in output nodes reach maximum connection strength in the self-evolution.

Need to find a better approach trimming connections.

**Good Find!** Breaking the self-loop two-node system seems removing the blind firing outputs shown above.
Now it';s turn to lift restrictions to allow some kind of firing in the system.
Maybe change the image input format? (From constant 1s to a flip between 0/1).

2026-05-01 v0.1:
An observation: 
Before the execution of manifold, the number of 1s logged in manifold is 50999. This represents the number of 1s in the input image.
However, after the 1st time execution, the number of 1s are dropped to 15465, this shouldn't happen because the 1s in the image should still **PERSIST** in the manifold, which is not the case.
Hence there is something wrong with the manifold. Need to investigate why the number of 1s are dropped.

----------
2026-05-01 v0.2:
After image loading fix, signal 1s persist in the manifold now. This is a very important advancement!!!
Next step is to update the feedback loop feature allowing the manifold to be evolved to the correct output behavior.

The manifold loading/saving feature is in place now.

2026-05-05 v0.3:
Temporarily stop using random ordering of global_array because it genereates too much noises.
Switch back to order loop, and start initializing global_array with random k and directions.
Create a simple scaffolding back propagation out of the hypercube manifold - With correct predictions, more image inputs can be consumed by the manifold.
The manifold starts generating outputs, even though it's wrong - This is a good sign that the internal shape of manifold can be changed by different patterns of inputs.
Next step is to find a good way of controlling the manifold shape evolvement.

2026-05-07 v0.4:
New changes on manifold initialization with a minimum strength & try to initialize a super-conductor manifold at very beginning and collapse it in following steps.
This mimics the quantum mechanics and the BigBang (super-conductor manifold in initialization)

2026-05-28 v0.5:
The node connections would reach maximum after the manifold evolvement starts. The signals are soaked among everywhere inside the manifold.
Need too find a way to constraint such explosion development.

2026-06-18 v1.0:
Hypercube printed out in Hammer String structure clearly shows that each different picture has a very different stable internal structure inside 
the hypercube. They each have a different fingerprint!