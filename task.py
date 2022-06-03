import sys
import enum
import numpy as np
import xml.etree.ElementTree as ET

PERCENTILES = [25, 50, 75, 90]

class ElementaryTask:

    def __init__(self, name, days) -> None:
        self.name = name
        self.days = days
        self._estimates = {}
        self._base_scale = 0.9

    @classmethod
    def from_xml(cls, xml):
        return cls(name=xml.tag, days=int(xml.attrib['days']))

    def sample(self, n_samples: int):
        result = np.random.gamma(
            shape=self.days+1,
            scale=self._base_scale + self.days / 100, 
            size=n_samples
        )

        for q in PERCENTILES:
            self._estimates[f'q{q}'] = str(round(np.percentile(result, q)))

        return result
    
    def to_xml(self):
        return ET.Element(self.name, days=str(self.days), **self._estimates)

class NoopTask:

    def __init__(self, name) -> None:
        self.name = name

    def sample(self, n_samples):
        return np.zeros(n_samples)

class Task:

    def __init__(self, name, days, prerequisites) -> None:
        self.name = name
        self.days = days
        self.own_task = NoopTask(name) if not days else ElementaryTask(name, days)
        self.prerequisites = list(prerequisites)
        self._estimates = {}

    @classmethod
    def from_xml(cls, xml):
        children = list(xml)
        if not children:
            return ElementaryTask.from_xml(xml)
        else:
            prereqs = [Task.from_xml(child) for child in children]
            days = None if 'days' not in xml.attrib else int(xml.attrib['days'])
            return cls(xml.tag, days, prerequisites=prereqs)

    def sample(self, n_samples):
        own_duration = self.own_task.sample(n_samples)

        prereq_samples = self._prereq_sample(n_samples)
        
        prereq_duration = np.max(prereq_samples, axis=1)

        result = own_duration + prereq_duration

        for q in PERCENTILES:
            self._estimates[f'q{q}'] = str(round(np.percentile(result, q)))

        return result

    def _prereq_sample(self, n_samples):
        prereq_samples = np.zeros((n_samples, len(self.prerequisites)))
        for i, prereq in enumerate(self.prerequisites):
            prereq_samples[:, i] = prereq.sample(n_samples)

        return prereq_samples

    def to_xml(self):
        element = \
            ET.Element(self.name, days=str(self.days), **self._estimates) if self.days else \
            ET.Element(self.name, **self._estimates)
        
        for req in self.prerequisites:
            element.append(req.to_xml())

        return element


def load(fin):
    return Task.from_xml(ET.fromstring(fin.read()))


if __name__ == '__main__':
    task = load(sys.stdin)
    task.sample(10000)

    xml = task.to_xml()
    ET.indent(xml, space=' '*4)
    print(ET.tostring(xml).decode())
