@echo off

FOR %%i IN (%*) DO ("%PCL_ROOT%pcl_obj2pcd_release.exe" %%i %%i.pcd)

REM pause