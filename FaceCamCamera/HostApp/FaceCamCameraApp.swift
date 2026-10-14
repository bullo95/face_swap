// App hôte minimale : installe / désinstalle l'extension caméra « FaceCam Camera ».
import SwiftUI
import SystemExtensions

@main
struct FaceCamCameraApp: App {
    var body: some Scene {
        WindowGroup("FaceCam Camera") {
            ContentView().frame(width: 460, height: 260)
        }.windowResizability(.contentSize)
    }
}

final class ExtensionManager: NSObject, ObservableObject, OSSystemExtensionRequestDelegate {
    @Published var status = "Extension non installée (ou état inconnu)."
    static let extensionID = "fr.bruno.facecamcamera.Extension"

    func activate() {
        status = "Installation demandée… macOS peut demander une autorisation dans Réglages Système."
        let req = OSSystemExtensionRequest.activationRequest(forExtensionWithIdentifier: Self.extensionID, queue: .main)
        req.delegate = self
        OSSystemExtensionManager.shared.submitRequest(req)
    }
    func deactivate() {
        status = "Désinstallation demandée…"
        let req = OSSystemExtensionRequest.deactivationRequest(forExtensionWithIdentifier: Self.extensionID, queue: .main)
        req.delegate = self
        OSSystemExtensionManager.shared.submitRequest(req)
    }
    func request(_ request: OSSystemExtensionRequest, actionForReplacingExtension existing: OSSystemExtensionProperties,
                 withExtension ext: OSSystemExtensionProperties) -> OSSystemExtensionRequest.ReplacementAction { .replace }
    func requestNeedsUserApproval(_ request: OSSystemExtensionRequest) {
        status = "Autorise l'extension dans Réglages Système › Général › Ouverture de session et extensions › Extensions de caméra."
    }
    func request(_ request: OSSystemExtensionRequest, didFinishWithResult result: OSSystemExtensionRequest.Result) {
        status = result == .completed ? "Extension installée : « FaceCam Camera » est disponible dans les applications vidéo."
                                      : "Terminé, un redémarrage peut être nécessaire."
    }
    func request(_ request: OSSystemExtensionRequest, didFailWithError error: Error) {
        status = "Échec : \(error.localizedDescription)"
    }
}

struct ContentView: View {
    @StateObject private var mgr = ExtensionManager()
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("FaceCam Camera").font(.title2).bold()
            Text("Caméra virtuelle pour FaceCam. Une fois installée, « FaceCam Camera » apparaît dans Zoom, Meet, FaceTime… et FaceCam y envoie son flux.")
                .font(.callout).foregroundStyle(.secondary)
            HStack {
                Button("Installer l'extension") { mgr.activate() }.buttonStyle(.borderedProminent)
                Button("Désinstaller") { mgr.deactivate() }
            }
            Text(mgr.status).font(.footnote).textSelection(.enabled)
            Spacer()
        }.padding(20)
    }
}
