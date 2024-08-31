#pragma once

class DeviceScene;
struct SceneGeometry;
struct Triangle;

class HostScene {
public:
	HostScene() = default;
	explicit HostScene(DeviceScene* device_scene);
	void syncDeviceGeometry();
	void AddTriangle(const Triangle& triangle);

private:
	DeviceScene* m_DeviceScene = nullptr;//non-owning; provided initially by renderer
};