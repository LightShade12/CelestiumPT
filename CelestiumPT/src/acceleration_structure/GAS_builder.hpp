#pragma once

class HostScene;

class GASBuilder {
public:
	GASBuilder() = default;

	void build(HostScene* host_scene);
	void refresh(HostScene* host_scene);
private:
};