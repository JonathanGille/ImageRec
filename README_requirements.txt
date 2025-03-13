zwei dateien sind zu groß für github. eine kopie liegt in seafile () zusammen mit einer textdatei wo die hinkopiert werden müssen.
am besten dann direkt bei github zu .gitignore hinzufügen, da die nicht gepusht werden können.
Nachtrag -> neues venv erstellt, evtl anderer pfad als in seafile steht (wahrscheinlich aber gleich), evtl downloaded 'pip install torch' das aber selber


VENV aktivieren:
.\.venv\Scripts\activate

beim Fehler: 
"C:\Users\jonat\Documents\GitHub\ImageRec\.venv\Scripts\Activate.ps1" kann nicht geladen werden, da die 
Ausführung von Skripts auf diesem System deaktiviert ist. Weitere Informationen finden Sie unter "about_Execution_Policies"
(https:/go.microsoft.com/fwlink/?LinkID=135170).
In Zeile:1 Zeichen:1
+ .\.venv\Scripts\activate
+ ~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : Sicherheitsfehler: (:) [], PSSecurityException
    + FullyQualifiedErrorId : UnauthorizedAccess

Lösung:
Für diese Sitzung: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
Für immer (für aktuellen Nutzer): Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned


VENV deaktivieren:
deactivate

Pytorch Cuda installieren:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

