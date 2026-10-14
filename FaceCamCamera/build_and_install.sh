#!/bin/zsh
# Construit l'app hôte « FaceCam Camera » (avec son extension caméra) et l'installe dans /Applications.
# Prérequis : compte Apple connecté dans Xcode (Réglages › Comptes), équipe 57LHMQV3Y5.
set -e
cd "$(dirname "$0")"
command -v xcodegen >/dev/null || brew install xcodegen
xcodegen generate >/dev/null
xcodebuild -project FaceCamCamera.xcodeproj -scheme FaceCamCamera -configuration Release \
  -derivedDataPath build -allowProvisioningUpdates build 2>&1 | grep -E "error:|BUILD" || true
APP="build/Build/Products/Release/FaceCam Camera.app"
[ -d "$APP" ] || { echo "Construction échouée"; exit 1; }
# l'activation d'une extension système exige que l'app hôte soit dans /Applications
rm -rf "/Applications/FaceCam Camera.app"
cp -R "$APP" /Applications/
echo "Installée : /Applications/FaceCam Camera.app — lance-la et clique « Installer l'extension »."
# client CoreMediaIO utilisé par FaceCam (Python)
(cd Client && swiftc -O -emit-library -o libfacecam_cmio.dylib -module-name facecam_cmio facecam_cmio.swift \
   -framework CoreMediaIO -framework CoreMedia -framework CoreVideo && echo "Client : Client/libfacecam_cmio.dylib")
