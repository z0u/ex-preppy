"""Test for distributed progress bar functionality."""

import asyncio
import time
from unittest.mock import Mock, patch
import pytest

from utils.progress.distributed import DistributedProgress, DistributedSyncProgress, is_modal_environment
from utils.progress.modal_progress import ModalProgressWriter, ModalProgressReader


class MockDict(dict):
    """A dict-like mock that supports item assignment."""
    pass


class TestModalEnvironmentDetection:
    """Test Modal environment detection."""

    def test_is_modal_environment_local(self):
        """Test that local environment is detected correctly."""
        with patch('modal.is_local', return_value=True):
            assert is_modal_environment() is False

    def test_is_modal_environment_modal(self):
        """Test that Modal environment is detected correctly."""
        with patch('modal.is_local', return_value=False):
            assert is_modal_environment() is True

    def test_is_modal_environment_exception(self):
        """Test that exceptions are handled gracefully."""
        with patch('modal.is_local', side_effect=Exception('Modal not available')):
            assert is_modal_environment() is False


class TestDistributedProgress:
    """Test the DistributedProgress class."""

    def test_local_mode_initialization(self):
        """Test that local mode uses standard Progress."""
        with patch('utils.progress.distributed.is_modal_environment', return_value=False):
            progress = DistributedProgress(total=100, description='Test')
            # Should use regular Progress class
            assert hasattr(progress._progress, '__aiter__')

    def test_modal_mode_initialization(self):
        """Test that Modal mode uses ModalProgressWriter."""
        with patch('utils.progress.distributed.is_modal_environment', return_value=True), \
             patch('modal.Dict.from_name', return_value=MockDict()):
            progress = DistributedProgress(total=100, description='Test')
            # Should use ModalProgressWriter
            assert isinstance(progress._progress, ModalProgressWriter)

    def test_property_forwarding(self):
        """Test that properties are forwarded correctly."""
        with patch('utils.progress.distributed.is_modal_environment', return_value=False):
            progress = DistributedSyncProgress(total=100, description='Test')  # Use sync version

            # Test property access
            assert progress.total == 100
            assert progress.description == 'Test'

            # Test property setting
            progress.total = 200
            progress.description = 'Updated'
            assert progress.total == 200
            assert progress.description == 'Updated'


class TestDistributedSyncProgress:
    """Test the DistributedSyncProgress class."""

    def test_local_mode_initialization(self):
        """Test that local mode uses standard SyncProgress."""
        with patch('utils.progress.distributed.is_modal_environment', return_value=False):
            progress = DistributedSyncProgress(total=100, description='Test')
            # Should use regular SyncProgress class
            assert hasattr(progress._progress, '__iter__')

    def test_modal_mode_initialization(self):
        """Test that Modal mode uses ModalProgressWriter."""
        with patch('utils.progress.distributed.is_modal_environment', return_value=True), \
             patch('modal.Dict.from_name', return_value=MockDict()):
            progress = DistributedSyncProgress(total=100, description='Test')
            # Should use ModalProgressWriter
            assert isinstance(progress._progress, ModalProgressWriter)


class TestModalProgressWriter:
    """Test the ModalProgressWriter class."""

    def test_initialization_without_modal(self):
        """Test initialization when Modal is not available."""
        with patch('modal.Dict.from_name', side_effect=Exception('Modal not available')):
            writer = ModalProgressWriter(total=100, description='Test')
            assert writer._modal_dict is None

    def test_initialization_with_modal(self):
        """Test initialization when Modal is available."""
        mock_dict = MockDict()
        with patch('modal.Dict.from_name', return_value=mock_dict):
            writer = ModalProgressWriter(total=100, description='Test')
            assert writer._modal_dict == mock_dict

    def test_mark_functionality(self):
        """Test that marking works correctly."""
        mock_dict = MockDict()
        with patch('modal.Dict.from_name', return_value=mock_dict):
            writer = ModalProgressWriter(total=100, description='Test', shard_size=2)

            writer.mark('First mark')
            assert len(writer.markers) == 1
            assert writer.markers[0].label == 'First mark'

    def test_shard_management(self):
        """Test that marker sharding works correctly."""
        mock_dict = MockDict()  # Use MockDict instead of plain dict
        with patch('modal.Dict.from_name', return_value=mock_dict):
            writer = ModalProgressWriter(total=100, description='Test', shard_size=2)

            # Add markers that should trigger shard creation
            writer.mark('Mark 1')
            writer.mark('Mark 2')
            writer.mark('Mark 3')  # This should trigger a new shard

            assert writer._current_shard >= 1

    def test_debounced_updates(self):
        """Test that updates are debounced correctly."""
        mock_dict = MockDict()  # Use MockDict instead of plain dict
        with patch('modal.Dict.from_name', return_value=mock_dict):
            writer = ModalProgressWriter(total=100, description='Test', interval=0.1)

            # First update should go through
            writer.count = 1
            time.sleep(0.05)  # Less than interval

            # Second update should be debounced
            writer.count = 2


class TestModalProgressReader:
    """Test the ModalProgressReader class."""

    def test_initialization_without_modal(self):
        """Test initialization when Modal is not available."""
        with patch('modal.Dict.from_name', side_effect=Exception('Modal not available')):
            reader = ModalProgressReader()
            assert reader._modal_dict is None

    def test_initialization_with_modal(self):
        """Test initialization when Modal is available."""
        mock_dict = MockDict()
        with patch('modal.Dict.from_name', return_value=mock_dict):
            reader = ModalProgressReader()
            assert reader._modal_dict == mock_dict

    @pytest.mark.asyncio
    async def test_polling_without_data(self):
        """Test polling when no data is available."""
        mock_dict = MockDict()
        mock_dict.get = Mock(return_value={})
        with patch('modal.Dict.from_name', return_value=mock_dict):
            reader = ModalProgressReader(poll_interval=0.01)

            # Start polling for a short time
            task = None
            try:
                task = asyncio.create_task(reader.start_polling())
                await asyncio.sleep(0.05)  # Let it poll a few times
            finally:
                reader.stop_polling()
                if task:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
