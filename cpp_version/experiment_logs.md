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