@echo off

FOR %%i IN (%*) DO ("C:\Program Files\PCL 1.9.1\bin\pcl_obj2pcd_release.exe" %%i %%i.pcd)

REM pause