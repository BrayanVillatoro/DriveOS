Set WshShell = CreateObject("WScript.Shell")
Set FSO = CreateObject("Scripting.FileSystemObject")

' Get the directory where this script is located
ScriptDir = FSO.GetParentFolderName(WScript.ScriptFullName)

' Check for virtual environment
If FSO.FileExists(ScriptDir & "\.venv\Scripts\pythonw.exe") Then
    WshShell.Run """" & ScriptDir & "\.venv\Scripts\pythonw.exe"" """ & ScriptDir & "\launch_gui.pyw""", 0, False
ElseIf FSO.FileExists(ScriptDir & "\.venv311\Scripts\pythonw.exe") Then
    WshShell.Run """" & ScriptDir & "\.venv311\Scripts\pythonw.exe"" """ & ScriptDir & "\launch_gui.pyw""", 0, False
Else
    MsgBox "Error: Python environment not found!" & vbCrLf & "Please run INSTALL.bat first.", vbCritical, "DriveOS Error"
End If
