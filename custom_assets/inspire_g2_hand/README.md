# Inspire G2 hand assets

These URDF and STL files were copied from the vendor package
`因时G2灵巧手URDF和SDK.zip`.

The only change to the URDF files is that ROS package mesh paths such as
`package://RH5DG2_R/meshes/...` were converted to relative paths
`meshes/...`, allowing Pinocchio and SAPIEN to load the model directly from
this self-contained project.

Each hand has 18 revolute joints. Five joints use URDF mimic tags, leaving
13 independent joints for retargeting.
