Set WshShell = CreateObject("WScript.Shell")
Set FSO = CreateObject("Scripting.FileSystemObject")

batPath = FSO.BuildPath(FSO.GetParentFolderName(WScript.ScriptFullName), "start_all.bat")
WshShell.Run Chr(34) & batPath & Chr(34), 0, False

Set FSO = Nothing
Set WshShell = Nothing