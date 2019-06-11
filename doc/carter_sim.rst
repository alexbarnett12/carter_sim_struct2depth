Getting Started With Carter Sim
===============================
This section describes how to run the :code:`carter_sim` application for simulation of Carter
navigation, mapping and joystick control.

Running carter_sim
------------------
To run the :code:`carter_sim` application in navigation mode, a map must be specified with a command
similar to the following:

.. code-block:: bash

   bob@desktop:~/isaac$ bazel run apps/carter_sim:carter_sim -- --config="apps/assets/maps/carter_warehouse_p.config.json" --graph="apps/assets/maps/carter_warehouse_p.graph.json"

Supported Maps for Navigation
-----------------------------
The :code:`carter_sim` application supports the carter_warehouse_p, carter_office, and hospital maps
in this release. Start the application with the command below that specifies the map you wish to
use.

For the carter_warehouse_p map, use the following command:

.. code-block:: bash

   bob@desktop:~/isaac$ bazel run apps/carter_sim:carter_sim -- --config="apps/assets/maps/carter_warehouse_p.config.json" --graph="apps/assets/maps/carter_warehouse_p.graph.json"

For the carter_office map, use the following command:

.. code-block:: bash

   bob@desktop:~/isaac$ bazel run apps/carter_sim:carter_sim -- --config="apps/assets/maps/carter_office.config.json" --graph="apps/assets/maps/carter_office.graph.json"

For the hospital map, use the following command:

.. code-block:: bash

   bob@desktop:~/isaac$ bazel run apps/carter_sim:carter_sim -- --config="apps/assets/maps/hospital.config.json" --graph="apps/assets/maps/hospital.graph.json"

The map name specified in the command line must match the same name in Isaac Sim. See Isaac Sim
documentation for where the robot is spawned in each of the different maps.


Joystick Control
----------------

To simulate any of the maps with joystick enabled, use the :code:`carter_sim_joystick` application
along with one of the map configuration/graph files described above. In the
:code:`carter_sim_joystick` application, press and hold the left shoulder button to enter manual
control mode, and then and use the left thumbstick to move the robot forward/backward and the right
thumbstick to turn left/right.

.. code-block:: bash

   bob@desktop:~/isaac$ bazel run apps/carter_sim:carter_sim_joystick -- --config="apps/assets/maps/carter_warehouse_p.config.json" --graph="apps/assets/maps/carter_warehouse_p.graph.json"

Mapping Mode
------------
To create new maps from simulation use the :code:`carter_sim_mapping` application. Maps are
automatically saved in /tmp when the application is closed.

.. code-block:: bash

   bob@desktop:~/isaac$ bazel run apps/carter_sim:carter_sim_mapping
