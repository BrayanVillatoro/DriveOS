Set oWS = WScript.CreateObject("WScript.Shell")
sLinkFile = oWS.SpecialFolders("Desktop") & "\DriveOS.lnk"
Set oLink = oWS.CreateShortcut(sLinkFile)
oLink.TargetPath = WScript.ScriptFullName
oLink.TargetPath = Replace(oLink.TargetPath, "create_shortcut.vbs", "DriveOS.bat")
oLink.WorkingDirectory = Replace(WScript.ScriptFullName, "\create_shortcut.vbs", "")
oLink.Description = "DriveOS Racing Line Analyzer"
oLink.IconLocation = "%SystemRoot%\System32\imageres.dll,99"
oLink.Save
WScript.Echo "Shortcut created on Desktop!"
