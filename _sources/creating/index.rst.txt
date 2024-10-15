Creating Merger Forests
=======================

Merger tree forests are created from the HACC treenode files, cf. [Rangel2020]_.
The code rearranges the snapshot-by-snapshot catalog format to a convenient and
efficient tree-format. Each processed tree-file is self-contained, i.e. no trees
are split among multiple files. See :ref:`sec-forest-layout` for more
information about the final product.

The MPI enabled python script in ``haccytrees/scripts/treenodes2forest.py`` is
the main executable which sets up the configuration and starts the conversion
routine contained in :meth:`haccytrees.mergertrees.assemble.catalog2tree`. The
settings have to be provided to the script through a configuration file, see
:ref:`sec-assemble-configuration` for an example.

Installing the ``haccytrees`` python module (see :ref:`install-haccytrees`) will
add ``haccytrees-convert`` to your PATH (which is a wrapper of
``treenodes2forest.py``). The code can then be executed like

.. code-block:: bash

   CONFIG=treenodes2forest.cfg
   mpirun -n <NPROC> haccytrees-convert ${CONFIG}

.. note::

   Alternatively, you can run
   ``mpirun -n <NPROC> python <PATH_TO_treenodes2forest.py> ${CONFIG}``

.. note::

   All treenode catalog files need to be located in a single folder. If
   ``MergerTrees_updated`` is being used, this folder usually does not contain
   the first snapshot. One strategy is to symlink all catalogs in
   ``MergerTrees_updated`` as well as the first snapshot from the
   ``MergerTrees`` folder into a new folder.

Routine Outline
---------------

The assembling of the merger forests proceeds in three main steps:

- distributing the halo catalogs among the MPI ranks such that each individual
  tree is fully contained on a rank.

- finding the correct ordering of the halos, corresponding to the layout
  described in :ref:`sec-forest-layout`.

- writing the forests to HDF5 files.

Distributing the data
^^^^^^^^^^^^^^^^^^^^^

The rank to which a tree is assigned is determined by the position of the root
halo at the last step, i.e. :math:`z=0`. The partitioning of the simulation
volume is determined by the :class:`mpipartition.Partition`
class, using MPI's 3D cartesian communicator.

We start by distributing the halos in the final snapshot, using the abstracted
distribution function :meth:`mpipartition.distribute`. We then
iterate backwards over the snapshots. The halos in each snapshot are first
distributed by their position. Afterwards, halos that may have crossed the rank
boundaries are accounted for by marking all halos that don't have a local
descendant halo. Those halos are then communicated with the 26 directly
neighboring ranks, using a MPI graph communicator connecting each 26
neighbor-set symmetrically. If there are still unaccounted halos left, those are
assigned using an ``all2all`` exchange. This exchange functionality is
implemented in :meth:`mpipartition.exchange`.

At each step, we also take note of the descendant halo array index in the
previous step. This information then simplifies the next step, the reordering of
the halos to form a self-contained tree.

Finding the halo ordering
^^^^^^^^^^^^^^^^^^^^^^^^^

After the reading and distributing phase, each rank now contains all the data it
needs to generate it's own self-contained forest. From the descendant index
stored during the previous phase we can then determine where in the final array
each halo has to go in order to obtain the required layout.

In a first step, we calculate the size of each subtree, starting from the
earliest snapshot. Halos that are leaves of the tree have size 1 by definition.
We can then iteratively add the halos size to its descendant in the next
snapshot. After processing the latest snapshot, we know the size of each
self-contained trees.

Knowing the size of each subtree, we can then determine the halo's position,
starting from the latest snapshot. Each root halo in the list is offset from its
previous halo by the size of that halos subtree. In the earlier snapshots, each
halo is positioned at its descendant halos position plus the subtree sizes of
the halos that have the same descendant and came earlier in the halo list. By
previously having ordered the halos in each snapshot by decreasing mass, halos
that have the same descendant halo are automatically ordered the same way.

After this step, we know at which array position every halo in the
snapshot-ordered catalog has to go during the next phase.

Writing the data
^^^^^^^^^^^^^^^^

In order to minimize the memory requirements, all the rank-local data of each
snapshot is stored into temporary containers (see :ref:`temporary-storage`). The
previous step, for example, only requires the descendant index to be in memory.
During the writing step, we now read the temporary files field-by-field, reorder
the data according to the previously determined halo ordering, and store that
field into an HDF5 dataset. By iterating over the fields individually, we only
need to keep one array in memory at a time. For the full Last Journey dataset
for example, 32 nodes were more than sufficient to process the forests.



.. _sec-assemble-configuration:

Configuration
-------------

The configuration file is in the `ini` style (cf. `configparser
<https://docs.python.org/3/library/configparser.html>`_) containing information
about the simulation, the data location, the field names in the treenode
catalogs and the forest output, as well as some switches to optimize the
routine. See the definition of the parameters in the following example
configuration:

.. literalinclude:: ../../haccytrees/scripts/treenodes2forest.example.cfg
   :caption: Example configuration for Last Journey
   :language: ini


Example: Last Journey
---------------------

Creating the :ref:`sim_LastJourney` Merger Forest took 3.2 hours on cooley,
using 32 nodes with 12 MPI ranks each (total of 384 ranks). The majority of the
time was spent in reading the treenode files (1.3 hours) and doing temporary IO
(1h) in order to keep the memory consumption in each rank low. A much smaller
fraction of the total time is in MPI communications (distribute and exchange),
~40min. Calculating the sub-tree sizes and data ordering took an
insignificant amount of time.


.. figure:: LJ_timing_32nodes.svg
   :width: 100%
   :align: center

   Processing time of Last Journey, split by task


Galaxy Merger trees
-------------------

The code also works for galaxy mergertrees. See
``haccytrees/scripts/treenodes2forest.galaxymergertree.example.cfg`` for an
example configuration file.

References
----------

.. autofunction:: haccytrees.mergertrees.assemble.catalog2tree


.. [Rangel2020] Rangel et al. (2020)
   arXiv:`2008.08519 <https://arxiv.org/abs/2008.08519>`_