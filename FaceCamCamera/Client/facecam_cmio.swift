// Client CoreMediaIO : pousse des images BGRA dans le flux d'entrée (sink) de « FaceCam Camera ».
// Compilé en dylib avec interface C (@_cdecl), appelé depuis Python via ctypes.
import Foundation
import CoreMediaIO
import CoreMedia
import CoreVideo

final class SinkClient {
    var device: CMIOObjectID = 0
    var sink: CMIOStreamID = 0
    var queue: CMSimpleQueue?
    var queueRef: Unmanaged<CMSimpleQueue>?
    var width = 0, height = 0
    var fdesc: CMFormatDescription?
    var pool: CVPixelBufferPool?
    var frameIndex: Int64 = 0

    static func propertyData<T>(_ obj: CMIOObjectID, _ selector: CMIOObjectPropertySelector, _ scope: CMIOObjectPropertyScope = .init(kCMIOObjectPropertyScopeGlobal), _ type: T.Type) -> [T] {
        var addr = CMIOObjectPropertyAddress(mSelector: selector, mScope: scope, mElement: .init(kCMIOObjectPropertyElementMain))
        var size: UInt32 = 0
        guard CMIOObjectGetPropertyDataSize(obj, &addr, 0, nil, &size) == 0, size > 0 else { return [] }
        let count = Int(size) / MemoryLayout<T>.size
        var out = [T](unsafeUninitializedCapacity: count) { buf, n in
            var used: UInt32 = 0
            _ = CMIOObjectGetPropertyData(obj, &addr, 0, nil, size, &used, buf.baseAddress!)
            n = Int(used) / MemoryLayout<T>.size
        }
        if out.count > count { out.removeLast(out.count - count) }
        return out
    }
    static func stringProperty(_ obj: CMIOObjectID, _ selector: CMIOObjectPropertySelector) -> String? {
        var addr = CMIOObjectPropertyAddress(mSelector: selector, mScope: .init(kCMIOObjectPropertyScopeGlobal), mElement: .init(kCMIOObjectPropertyElementMain))
        var size: UInt32 = UInt32(MemoryLayout<CFString?>.size)
        var value: Unmanaged<CFString>?
        var used: UInt32 = 0
        guard CMIOObjectGetPropertyData(obj, &addr, 0, nil, size, &used, &value) == 0, let v = value else { return nil }
        return v.takeRetainedValue() as String
    }

    func open(deviceUID: String, width: Int, height: Int) -> Int32 {
        // les périphériques d'extensions CMIO sont visibles sans réglage particulier ; on active quand même la DAL
        var allow: UInt32 = 1
        var allowAddr = CMIOObjectPropertyAddress(mSelector: .init(kCMIOHardwarePropertyAllowScreenCaptureDevices), mScope: .init(kCMIOObjectPropertyScopeGlobal), mElement: .init(kCMIOObjectPropertyElementMain))
        CMIOObjectSetPropertyData(.init(kCMIOObjectSystemObject), &allowAddr, 0, nil, UInt32(MemoryLayout<UInt32>.size), &allow)
        let devices: [CMIOObjectID] = Self.propertyData(.init(kCMIOObjectSystemObject), .init(kCMIOHardwarePropertyDevices), .init(kCMIOObjectPropertyScopeGlobal), CMIOObjectID.self)
        guard let dev = devices.first(where: { Self.stringProperty($0, .init(kCMIODevicePropertyDeviceUID)) == deviceUID }) else { return -1 }
        device = dev
        let streams: [CMIOStreamID] = Self.propertyData(dev, .init(kCMIODevicePropertyStreams), .init(kCMIODevicePropertyScopeInput), CMIOStreamID.self)
            + Self.propertyData(dev, .init(kCMIODevicePropertyStreams), .init(kCMIOObjectPropertyScopeGlobal), CMIOStreamID.self)
        // le flux « sink » a la direction 1 (entrée vers le périphérique)
        var sinkID: CMIOStreamID = 0
        for s in Set(streams) {
            let dir: [UInt32] = Self.propertyData(s, .init(kCMIOStreamPropertyDirection), .init(kCMIOObjectPropertyScopeGlobal), UInt32.self)
            if dir.first == 1 { sinkID = s; break }
        }
        guard sinkID != 0 else { return -2 }
        sink = sinkID
        var q: Unmanaged<CMSimpleQueue>?
        let st = CMIOStreamCopyBufferQueue(sinkID, { _, _, _ in }, nil, &q)
        guard st == 0, let q else { return -3 }
        queueRef = q
        queue = q.takeUnretainedValue()
        self.width = width; self.height = height
        CMVideoFormatDescriptionCreate(allocator: kCFAllocatorDefault, codecType: kCVPixelFormatType_32BGRA, width: Int32(width), height: Int32(height), extensions: nil, formatDescriptionOut: &fdesc)
        let attrs: [String: Any] = [kCVPixelBufferWidthKey as String: width, kCVPixelBufferHeightKey as String: height,
                                    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                                    kCVPixelBufferIOSurfacePropertiesKey as String: [:]]
        CVPixelBufferPoolCreate(kCFAllocatorDefault, nil, attrs as CFDictionary, &pool)
        guard CMIODeviceStartStream(dev, sinkID) == 0 else { return -4 }
        return 0
    }

    func push(_ bgra: UnsafePointer<UInt8>, bytesPerRow: Int) -> Int32 {
        guard let queue, let pool, let fdesc else { return -10 }
        if CMSimpleQueueGetCount(queue) >= CMSimpleQueueGetCapacity(queue) { return 1 } // file pleine : image ignorée
        var pb: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &pb)
        guard let pb else { return -11 }
        CVPixelBufferLockBaseAddress(pb, [])
        if let dst = CVPixelBufferGetBaseAddress(pb) {
            let dstRow = CVPixelBufferGetBytesPerRow(pb)
            let rowBytes = width * 4
            for y in 0..<height {
                memcpy(dst + y * dstRow, bgra + y * bytesPerRow, rowBytes)
            }
        }
        CVPixelBufferUnlockBaseAddress(pb, [])
        let now = CMClockGetTime(CMClockGetHostTimeClock())
        var timing = CMSampleTimingInfo(duration: CMTime(value: 1, timescale: 30), presentationTimeStamp: now, decodeTimeStamp: .invalid)
        var sbuf: CMSampleBuffer?
        CMSampleBufferCreateForImageBuffer(allocator: kCFAllocatorDefault, imageBuffer: pb, dataReady: true, makeDataReadyCallback: nil, refcon: nil,
                                           formatDescription: fdesc, sampleTiming: &timing, sampleBufferOut: &sbuf)
        guard let sbuf else { return -12 }
        frameIndex += 1
        let st = CMSimpleQueueEnqueue(queue, element: Unmanaged.passRetained(sbuf).toOpaque())
        return st == 0 ? 0 : -13
    }

    func close() {
        if device != 0 && sink != 0 { CMIODeviceStopStream(device, sink) }
        queueRef?.release(); queueRef = nil; queue = nil
        device = 0; sink = 0
    }
}

private var clients: [Int32: SinkClient] = [:]
private var nextHandle: Int32 = 1

@_cdecl("facecam_cmio_open")
public func facecam_cmio_open(_ uid: UnsafePointer<CChar>, _ width: Int32, _ height: Int32) -> Int32 {
    let c = SinkClient()
    let st = c.open(deviceUID: String(cString: uid), width: Int(width), height: Int(height))
    if st != 0 { return st }
    let h = nextHandle; nextHandle += 1; clients[h] = c
    return h
}
@_cdecl("facecam_cmio_push")
public func facecam_cmio_push(_ handle: Int32, _ bgra: UnsafePointer<UInt8>, _ bytesPerRow: Int32) -> Int32 {
    guard let c = clients[handle] else { return -20 }
    return c.push(bgra, bytesPerRow: Int(bytesPerRow))
}
@_cdecl("facecam_cmio_close")
public func facecam_cmio_close(_ handle: Int32) {
    clients[handle]?.close(); clients[handle] = nil
}
@_cdecl("facecam_cmio_list")
public func facecam_cmio_list(_ out: UnsafeMutablePointer<CChar>, _ cap: Int32) -> Int32 {
    let devices: [CMIOObjectID] = SinkClient.propertyData(.init(kCMIOObjectSystemObject), .init(kCMIOHardwarePropertyDevices), .init(kCMIOObjectPropertyScopeGlobal), CMIOObjectID.self)
    let s = devices.map { "\(SinkClient.stringProperty($0, .init(kCMIODevicePropertyDeviceUID)) ?? "?")|\(SinkClient.stringProperty($0, .init(kCMIOObjectPropertyName)) ?? "?")" }.joined(separator: "\n")
    let bytes = Array(s.utf8CString)
    let n = min(bytes.count, Int(cap) - 1)
    for i in 0..<n { out[i] = bytes[i] }
    out[n] = 0
    return Int32(n)
}
