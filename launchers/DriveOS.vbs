Set WshShell = CreateObject("WScript.Shell")
Set FSO = CreateObject("Scripting.FileSystemObject")

' Get the parent directory (project root)
ScriptDir = FSO.GetParentFolderName(WScript.ScriptFullName)
RootDir = FSO.GetParentFolderName(ScriptDir)

' Check for virtual environment
If FSO.FileExists(RootDir & "\.venv\Scripts\pythonw.exe") Then
    WshShell.Run """" & RootDir & "\.venv\Scripts\pythonw.exe"" """ & ScriptDir & "\launch_gui.pyw""", 0, False
ElseIf FSO.FileExists(RootDir & "\.venv311\Scripts\pythonw.exe") Then
    WshShell.Run """" & RootDir & "\.venv311\Scripts\pythonw.exe"" """ & ScriptDir & "\launch_gui.pyw""", 0, False
Else
    MsgBox "Error: Python environment not found!" & vbCrLf & "Please run INSTALL.bat first.", vbCritical, "DriveOS Error"
End If
