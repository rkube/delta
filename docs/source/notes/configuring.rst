***********************************
Configuring data analysis workflows
***********************************

Delta's individual software components are configured by dictionaries.
The reference `processor` loads the configuration from a json file and
expects multiple objects under the root node. These are used to configure
the individual components as follows:

+------------------------+-----------------------+
| Component              | Object name           |
+========================+=======================+
| :ref:`Data sources`    | diagnostic            |
+------------------------+-----------------------+
| :ref:`Data models`     | diagnostic            |
+------------------------+-----------------------+
| :ref:`Streaming`       | transport             |
+------------------------+-----------------------+
| :ref:`Pre-processing`  | preprocess            |
+------------------------+-----------------------+
| :ref:`Data analysis`   | analysis              |
+------------------------+-----------------------+
| :ref:`Storage`         | storage               |
+------------------------+-----------------------+