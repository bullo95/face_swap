import Foundation
import CoreMediaIO

let providerSource = FaceCamProviderSource(clientQueue: nil)
CMIOExtensionProvider.startService(provider: providerSource.provider)
CFRunLoopRun()
