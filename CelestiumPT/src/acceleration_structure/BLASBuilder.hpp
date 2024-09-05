#pragma once

class HostScene;

class BLASBuilder
{
public:

	BLASBuilder() = default;

	void build(HostScene* hscene);
};