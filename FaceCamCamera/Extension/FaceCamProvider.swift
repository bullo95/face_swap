// Extension caméra CoreMediaIO « FaceCam Camera ».
// Deux flux sur un même périphérique :
//  - sortie (source) : ce que voient Zoom, Meet, FaceTime… ;
//  - entrée (sink)   : FaceCam y pousse ses images (CMIOStreamCopyBufferQueue côté client).
// Sans client sur l'entrée, la sortie diffuse une mire sombre avec le texte « FaceCam ».
import Foundation
import CoreMediaIO
import CoreVideo
import CoreGraphics
import os.log

let log = Logger(subsystem: "fr.bruno.facecamcamera", category: "extension")

let kFrameRate = 30.0
let kWidth = 1280
let kHeight = 720
let kDeviceUID = "7A3F5C21-FACE-4CAM-B0D1-000000000001"

// MARK: - Provider

class FaceCamProviderSource: NSObject, CMIOExtensionProviderSource {
    private(set) var provider: CMIOExtensionProvider!
    private var deviceSource: FaceCamDeviceSource!

    init(clientQueue: DispatchQueue?) {
        super.init()
        provider = CMIOExtensionProvider(source: self, clientQueue: clientQueue)
        deviceSource = FaceCamDeviceSource(localizedName: "FaceCam Camera")
        do { try provider.addDevice(deviceSource.device) } catch { fatalError("addDevice: \(error)") }
    }
    func connect(to client: CMIOExtensionClient) throws {}
    func disconnect(from client: CMIOExtensionClient) {}
    var availableProperties: Set<CMIOExtensionProperty> { [.providerManufacturer, .providerName] }
    func providerProperties(forProperties properties: Set<CMIOExtensionProperty>) throws -> CMIOExtensionProviderProperties {
        let p = CMIOExtensionProviderProperties(dictionary: [:])
        if properties.contains(.providerManufacturer) { p.manufacturer = "FaceCam" }
        if properties.contains(.providerName) { p.name = "FaceCam Camera" }
        return p
    }
    func setProviderProperties(_ providerProperties: CMIOExtensionProviderProperties) throws {}
}

// MARK: - Device

class FaceCamDeviceSource: NSObject, CMIOExtensionDeviceSource {
    private(set) var device: CMIOExtensionDevice!
    private var sourceStream: FaceCamSourceStream!
    private var sinkStream: FaceCamSinkStream!
    private let videoDescription: CMFormatDescription
    private let format: CMIOExtensionStreamFormat

    init(localizedName: String) {
        var desc: CMFormatDescription?
        CMVideoFormatDescriptionCreate(allocator: kCFAllocatorDefault, codecType: kCVPixelFormatType_32BGRA,
                                       width: Int32(kWidth), height: Int32(kHeight), extensions: nil, formatDescriptionOut: &desc)
        videoDescription = desc!
        format = CMIOExtensionStreamFormat(formatDescription: videoDescription,
                                           maxFrameDuration: CMTime(value: 1, timescale: Int32(kFrameRate)),
                                           minFrameDuration: CMTime(value: 1, timescale: Int32(kFrameRate)),
                                           validFrameDurations: nil)
        super.init()
        device = CMIOExtensionDevice(localizedName: localizedName, deviceID: UUID(uuidString: kDeviceUID)!,
                                     legacyDeviceID: kDeviceUID, source: self)
        sourceStream = FaceCamSourceStream(localizedName: "FaceCam Camera", streamID: UUID(), streamFormat: format, device: device)
        sinkStream = FaceCamSinkStream(localizedName: "FaceCam Camera Input", streamID: UUID(), streamFormat: format, device: device)
        sourceStream.sink = sinkStream
        sinkStream.source = sourceStream
        do {
            try device.addStream(sourceStream.stream)
            try device.addStream(sinkStream.stream)
        } catch { fatalError("addStream: \(error)") }
    }
    var availableProperties: Set<CMIOExtensionProperty> { [.deviceTransportType, .deviceModel] }
    func deviceProperties(forProperties properties: Set<CMIOExtensionProperty>) throws -> CMIOExtensionDeviceProperties {
        let p = CMIOExtensionDeviceProperties(dictionary: [:])
        if properties.contains(.deviceTransportType) { p.transportType = kIOAudioDeviceTransportTypeVirtual }
        if properties.contains(.deviceModel) { p.model = "FaceCam Camera" }
        return p
    }
    func setDeviceProperties(_ deviceProperties: CMIOExtensionDeviceProperties) throws {}
}

// MARK: - Flux de sortie (vu par les applications)

class FaceCamSourceStream: NSObject, CMIOExtensionStreamSource {
    private(set) var stream: CMIOExtensionStream!
    let device: CMIOExtensionDevice
    private let streamFormat: CMIOExtensionStreamFormat
    weak var sink: FaceCamSinkStream?
    private var timer: DispatchSourceTimer?
    private let timerQueue = DispatchQueue(label: "fr.bruno.facecamcamera.timer", qos: .userInteractive)
    private var pool: CVPixelBufferPool?
    private var clientCount = 0
    var streaming: Bool { clientCount > 0 }

    init(localizedName: String, streamID: UUID, streamFormat: CMIOExtensionStreamFormat, device: CMIOExtensionDevice) {
        self.device = device
        self.streamFormat = streamFormat
        super.init()
        stream = CMIOExtensionStream(localizedName: localizedName, streamID: streamID, direction: .source, clockType: .hostTime, source: self)
        let attrs: [String: Any] = [kCVPixelBufferWidthKey as String: kWidth, kCVPixelBufferHeightKey as String: kHeight,
                                    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                                    kCVPixelBufferIOSurfacePropertiesKey as String: [:]]
        CVPixelBufferPoolCreate(kCFAllocatorDefault, nil, attrs as CFDictionary, &pool)
    }
    var formats: [CMIOExtensionStreamFormat] { [streamFormat] }
    var activeFormatIndex: Int = 0 { didSet { if activeFormatIndex != 0 { activeFormatIndex = 0 } } }
    var availableProperties: Set<CMIOExtensionProperty> { [.streamActiveFormatIndex, .streamFrameDuration] }
    func streamProperties(forProperties properties: Set<CMIOExtensionProperty>) throws -> CMIOExtensionStreamProperties {
        let p = CMIOExtensionStreamProperties(dictionary: [:])
        if properties.contains(.streamActiveFormatIndex) { p.activeFormatIndex = 0 }
        if properties.contains(.streamFrameDuration) { p.frameDuration = CMTime(value: 1, timescale: Int32(kFrameRate)) }
        return p
    }
    func setStreamProperties(_ streamProperties: CMIOExtensionStreamProperties) throws {}
    func authorizedToStartStream(for client: CMIOExtensionClient) -> Bool { true }

    func startStream() throws {
        clientCount += 1
        guard timer == nil else { return }
        let t = DispatchSource.makeTimerSource(flags: .strict, queue: timerQueue)
        t.schedule(deadline: .now(), repeating: 1.0 / kFrameRate, leeway: .milliseconds(2))
        t.setEventHandler { [weak self] in self?.tick() }
        t.resume()
        timer = t
        log.info("source: start")
    }
    func stopStream() throws {
        clientCount = max(0, clientCount - 1)
        if clientCount == 0 { timer?.cancel(); timer = nil; log.info("source: stop") }
    }

    /// Image reçue du flux d'entrée : relayée telle quelle (le client FaceCam envoie déjà 1280x720 BGRA).
    func send(fromSink sbuf: CMSampleBuffer) {
        lastSinkFrame = Date()
        stream.send(sbuf, discontinuity: [], hostTimeInNanoseconds: UInt64(CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sbuf)) * 1e9))
    }
    private var lastSinkFrame = Date.distantPast

    /// Sans client d'entrée depuis 0,5 s : mire de remplacement.
    private func tick() {
        guard Date().timeIntervalSince(lastSinkFrame) > 0.5, let pool else { return }
        var pb: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &pb)
        guard let pb else { return }
        CVPixelBufferLockBaseAddress(pb, [])
        if let base = CVPixelBufferGetBaseAddress(pb),
           let ctx = CGContext(data: base, width: kWidth, height: kHeight, bitsPerComponent: 8,
                               bytesPerRow: CVPixelBufferGetBytesPerRow(pb), space: CGColorSpaceCreateDeviceRGB(),
                               bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue) {
            ctx.setFillColor(CGColor(red: 0.08, green: 0.08, blue: 0.1, alpha: 1))
            ctx.fill(CGRect(x: 0, y: 0, width: kWidth, height: kHeight))
            ctx.setFillColor(CGColor(red: 0.35, green: 0.35, blue: 0.4, alpha: 1))
            ctx.fill(CGRect(x: kWidth/2 - 160, y: kHeight/2 - 4, width: 320, height: 8))
        }
        CVPixelBufferUnlockBaseAddress(pb, [])
        var fdesc: CMFormatDescription?
        CMVideoFormatDescriptionCreateForImageBuffer(allocator: kCFAllocatorDefault, imageBuffer: pb, formatDescriptionOut: &fdesc)
        guard let fdesc else { return }
        let now = CMClockGetTime(CMClockGetHostTimeClock())
        var timing = CMSampleTimingInfo(duration: CMTime(value: 1, timescale: Int32(kFrameRate)), presentationTimeStamp: now, decodeTimeStamp: .invalid)
        var sbuf: CMSampleBuffer?
        CMSampleBufferCreateForImageBuffer(allocator: kCFAllocatorDefault, imageBuffer: pb, dataReady: true, makeDataReadyCallback: nil,
                                           refcon: nil, formatDescription: fdesc, sampleTiming: &timing, sampleBufferOut: &sbuf)
        if let sbuf { stream.send(sbuf, discontinuity: [], hostTimeInNanoseconds: UInt64(now.seconds * 1e9)) }
    }
}

// MARK: - Flux d'entrée (sink) alimenté par FaceCam

class FaceCamSinkStream: NSObject, CMIOExtensionStreamSource {
    private(set) var stream: CMIOExtensionStream!
    let device: CMIOExtensionDevice
    private let streamFormat: CMIOExtensionStreamFormat
    weak var source: FaceCamSourceStream?
    private var client: CMIOExtensionClient?
    private var consuming = false

    init(localizedName: String, streamID: UUID, streamFormat: CMIOExtensionStreamFormat, device: CMIOExtensionDevice) {
        self.device = device
        self.streamFormat = streamFormat
        super.init()
        stream = CMIOExtensionStream(localizedName: localizedName, streamID: streamID, direction: .sink, clockType: .hostTime, source: self)
    }
    var formats: [CMIOExtensionStreamFormat] { [streamFormat] }
    var activeFormatIndex: Int = 0
    var availableProperties: Set<CMIOExtensionProperty> {
        [.streamActiveFormatIndex, .streamFrameDuration, .streamSinkBufferQueueSize, .streamSinkBuffersRequiredForStartup,
         .streamSinkBufferUnderrunCount, .streamSinkEndOfData]
    }
    func streamProperties(forProperties properties: Set<CMIOExtensionProperty>) throws -> CMIOExtensionStreamProperties {
        let p = CMIOExtensionStreamProperties(dictionary: [:])
        if properties.contains(.streamActiveFormatIndex) { p.activeFormatIndex = 0 }
        if properties.contains(.streamFrameDuration) { p.frameDuration = CMTime(value: 1, timescale: Int32(kFrameRate)) }
        if properties.contains(.streamSinkBufferQueueSize) { p.sinkBufferQueueSize = 8 }
        if properties.contains(.streamSinkBuffersRequiredForStartup) { p.sinkBuffersRequiredForStartup = 1 }
        if properties.contains(.streamSinkBufferUnderrunCount) { p.sinkBufferUnderrunCount = 0 }
        if properties.contains(.streamSinkEndOfData) { p.sinkEndOfData = false }
        return p
    }
    func setStreamProperties(_ streamProperties: CMIOExtensionStreamProperties) throws {}
    func authorizedToStartStream(for client: CMIOExtensionClient) -> Bool { self.client = client; return true }
    func startStream() throws { consuming = true; log.info("sink: start"); consumeNext() }
    func stopStream() throws { consuming = false; log.info("sink: stop") }

    private func consumeNext() {
        guard consuming, let client else { return }
        stream.consumeSampleBuffer(from: client) { [weak self] sbuf, _, _, hasMore, err in
            guard let self else { return }
            if let sbuf { self.source?.send(fromSink: sbuf) }
            if let err { log.error("sink consume: \(err.localizedDescription)") }
            if hasMore { self.consumeNext() } else {
                // attendre l'image suivante (notification de disponibilité)
                DispatchQueue.global(qos: .userInteractive).asyncAfter(deadline: .now() + .milliseconds(5)) { self.consumeNext() }
            }
        }
    }
}
