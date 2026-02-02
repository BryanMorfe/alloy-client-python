from __future__ import annotations

from typing import (
    Any,
    Dict,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Union,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)
from PIL.Image import Image

JsonSchemaValue = Dict[str, Any]

class SubscriptableBaseModel(BaseModel):
    def __getitem__(self, key: str) -> Any:
        """
        >>> msg = Message(role='user')
        >>> msg['role']
        'user'
        >>> msg = Message(role='user')
        >>> msg['nonexistent']
        Traceback (most recent call last):
        KeyError: 'nonexistent'
        """
        if key in self:
            return getattr(self, key)

        raise KeyError(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """
        >>> msg = Message(role='user')
        >>> msg['role'] = 'assistant'
        >>> msg['role']
        'assistant'
        >>> tool_call = Message.ToolCall(function=Message.ToolCall.Function(name='foo', arguments={}))
        >>> msg = Message(role='user', content='hello')
        >>> msg['tool_calls'] = [tool_call]
        >>> msg['tool_calls'][0]['function']['name']
        'foo'
        """
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        """
        >>> msg = Message(role='user')
        >>> 'nonexistent' in msg
        False
        >>> 'role' in msg
        True
        >>> 'content' in msg
        False
        >>> msg.content = 'hello!'
        >>> 'content' in msg
        True
        >>> msg = Message(role='user', content='hello!')
        >>> 'content' in msg
        True
        >>> 'tool_calls' in msg
        False
        >>> msg['tool_calls'] = []
        >>> 'tool_calls' in msg
        True
        >>> msg['tool_calls'] = [Message.ToolCall(function=Message.ToolCall.Function(name='foo', arguments={}))]
        >>> 'tool_calls' in msg
        True
        >>> msg['tool_calls'] = None
        >>> 'tool_calls' in msg
        True
        >>> tool = Tool()
        >>> 'type' in tool
        True
        """
        if key in self.model_fields_set:
            return True

        if value := self.__class__.model_fields.get(key):
            return value.default is not None

        return False

    def get(self, key: str, default: Any = None) -> Any:
        """
        >>> msg = Message(role='user')
        >>> msg.get('role')
        'user'
        >>> msg = Message(role='user')
        >>> msg.get('nonexistent')
        >>> msg = Message(role='user')
        >>> msg.get('nonexistent', 'default')
        'default'
        >>> msg = Message(role='user', tool_calls=[ Message.ToolCall(function=Message.ToolCall.Function(name='foo', arguments={}))])
        >>> msg.get('tool_calls')[0]['function']['name']
        'foo'
        """
        return getattr(self, key) if hasattr(self, key) else default

class Message(SubscriptableBaseModel):
    """
    Chat message.
    """

    role: str
    "Assumed role of the message. Response messages has role 'assistant' or 'tool'."

    content: Optional[str] = None
    'Content of the message. Response messages contains message fragments when streaming.'

    thinking: Optional[str] = None
    'Thinking content. Only present when thinking is enabled.'

    images: Optional[Sequence[Image]] = None
    """
    Optional list of image data for multimodal models.

    Valid input types are:

    - `str` or path-like object: path to image file
    - `bytes` or bytes-like object: raw image data

    Valid image formats depend on the model. See the model card for more information.
    """

    tool_name: Optional[str] = None
    'Name of the executed tool.'

    class ToolCall(SubscriptableBaseModel):
        """
        Model tool calls.
        """

        class Function(SubscriptableBaseModel):
            """
            Tool call function.
            """

            name: str
            'Name of the function.'

            arguments: Mapping[str, Any]
            'Arguments of the function.'

        function: Function
        'Function to be called.'

    tool_calls: Optional[Sequence[ToolCall]] = None
    """
    Tools calls to be made by the model.
    """


class Tool(SubscriptableBaseModel):
  type: Optional[str] = 'function'

  class Function(SubscriptableBaseModel):
    name: Optional[str] = None
    description: Optional[str] = None

    class Parameters(SubscriptableBaseModel):
      model_config = ConfigDict(populate_by_name=True)
      type: Optional[Literal['object']] = 'object'
      defs: Optional[Any] = Field(None, alias='$defs')
      items: Optional[Any] = None
      required: Optional[Sequence[str]] = None

      class Property(SubscriptableBaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        type: Optional[Union[str, Sequence[str]]] = None
        items: Optional[Any] = None
        description: Optional[str] = None
        enum: Optional[Sequence[Any]] = None

      properties: Optional[Mapping[str, Property]] = None

    parameters: Optional[Parameters] = None

  function: Optional[Function] = None


class BaseGenerateResponse(SubscriptableBaseModel):
    model: Optional[str] = None
    'Model used to generate response.'

    created_at: Optional[str] = None
    'Time when the request was created.'

    done: Optional[bool] = None
    'True if response is complete, otherwise False. Useful for streaming to detect the final response.'

    done_reason: Optional[str] = None
    'Reason for completion. Only present when done is True.'

    total_duration: Optional[int] = None
    'Total duration in nanoseconds.'

    load_duration: Optional[int] = None
    'Load duration in nanoseconds.'

    prompt_eval_count: Optional[int] = None
    'Number of tokens evaluated in the prompt.'

    prompt_eval_duration: Optional[int] = None
    'Duration of evaluating the prompt in nanoseconds.'

    eval_count: Optional[int] = None
    'Number of tokens evaluated in inference.'

    eval_duration: Optional[int] = None
    'Duration of evaluating inference in nanoseconds.'

class EmbedResponse(BaseGenerateResponse):
    """
    Response returned by embed requests.
    """

    embeddings: Sequence[Sequence[float]]
    'Embeddings of the inputs.'

class ChatResponse(BaseGenerateResponse):
    """
    Response returned by chat requests.
    """

    message: Message
    'Response message.'
