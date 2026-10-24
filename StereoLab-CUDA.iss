; Separate per-user CUDA edition. Keeps the DirectML edition independently installable.
#ifndef AppVersion
#define AppVersion "0.2.0-beta.6"
#endif
#ifndef PackageDir
#define PackageDir ".runtime\windows-cuda\dist\StereoLab"
#endif

[Setup]
AppId={{B975A8C2-9F2B-4E37-AC2F-8BD7DB857A83}
AppName=StereoLab CUDA
AppVersion={#AppVersion}
AppPublisher=Boss720
DefaultDirName={localappdata}\Programs\StereoLab CUDA
DefaultGroupName=StereoLab CUDA
PrivilegesRequired=lowest
DisableProgramGroupPage=yes
OutputBaseFilename=StereoLab-CUDA-{#AppVersion}-Setup
OutputDir=.runtime\windows-cuda\installer
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
Uninstallable=yes
UninstallDisplayName=StereoLab CUDA

[Files]
Source: "{#PackageDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\StereoLab CUDA\Start StereoLab CUDA"; Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\Start-StereoLab.ps1"""; WorkingDir: "{app}"; IconFilename: "{app}\StereoLab.exe"
Name: "{autoprograms}\StereoLab CUDA\Open browser (if StereoLab is running)"; Filename: "http://127.0.0.1:8765/"
Name: "{autoprograms}\StereoLab CUDA\Uninstall StereoLab CUDA"; Filename: "{uninstallexe}"
