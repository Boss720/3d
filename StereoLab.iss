; Local per-user installer for the StereoLab beta.
; Third-party notice and redistribution review must be completed before public release.
#ifndef AppVersion
#define AppVersion "0.2.0-beta.6"
#endif
#ifndef PackageDir
#define PackageDir "dist"
#endif

[Setup]
AppId={{B975A8C2-9F2B-4E37-AC2F-8BD7DB857A82}
AppName=StereoLab
AppVersion={#AppVersion}
AppPublisher=Boss720
DefaultDirName={localappdata}\Programs\StereoLab
DefaultGroupName=StereoLab
PrivilegesRequired=lowest
DisableProgramGroupPage=yes
OutputBaseFilename=StereoLab-{#AppVersion}-Setup
OutputDir=.runtime\stereolab-installer
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
Uninstallable=yes
UninstallDisplayName=StereoLab

[Files]
Source: "{#PackageDir}\StereoLab.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#PackageDir}\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#PackageDir}\Start-StereoLab.ps1"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\StereoLab\Start StereoLab"; Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\Start-StereoLab.ps1"""; WorkingDir: "{app}"; IconFilename: "{app}\StereoLab.exe"
Name: "{autoprograms}\StereoLab\Open browser (if StereoLab is running)"; Filename: "http://127.0.0.1:8765/"
Name: "{autoprograms}\StereoLab\Uninstall StereoLab"; Filename: "{uninstallexe}"

