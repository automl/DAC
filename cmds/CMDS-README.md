# CMDS
Contains commands used to run sigmoid experiments on a slurm cluster.
* All 1D-DIST commands train on 1D Sigmoids with binary action space where instances are sampled from a distribution.
* ALL 1D-INST commands train on 1D Sigmoids with binary action space on 100 fixed training instances and are evaluated on 100 unseen test instances.
* All *n*D*m*3 commands train on *n*D Sigmoids with *m*-ary action spaces where instances are sampled from a distribution. 