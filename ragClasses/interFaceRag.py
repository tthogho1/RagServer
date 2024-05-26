import abc


class Rag(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def setUrl(self, urlString: str) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def loadUrl(self) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def rag_chain(self, question: str) -> str:
        raise NotImplementedError()