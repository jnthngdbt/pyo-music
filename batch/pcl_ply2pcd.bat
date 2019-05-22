@echo off

echo %1
echo %~nf1.pcd
"C:\Program Files\PCL 1.6.0\bin\pcl_ply2pcd_release.exe" %1 %~nf1.pcd
REM echo %0
REM @echo %n1

pause