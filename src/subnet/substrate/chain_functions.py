from typing import Any, Optional
from substrateinterface import SubstrateInterface, Keypair, ExtrinsicReceipt
from substrateinterface.exceptions import SubstrateRequestException
from tenacity import retry, stop_after_attempt, wait_exponential, wait_fixed
from subnet.substrate.config import BLOCK_SECS
from tenacity import RetryCallState

retry_counter = 0

def increment_counter(retry_state: RetryCallState):
    global retry_counter
    retry_counter += 1
    print(f"Retry {retry_counter}: {retry_state}")


def get_block_number(substrate: SubstrateInterface):
  # @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(4))
  @retry(wait=wait_fixed(BLOCK_SECS+1), stop=stop_after_attempt(4))
  def make_query():
    try:
      with substrate as _substrate:
        block_hash = _substrate.get_block_hash()
        block_number = _substrate.get_block_number(block_hash)
        return block_number
    except SubstrateRequestException as e:
      print("Failed to get query request: {}".format(e))

  return make_query()

def validate(
  substrate: SubstrateInterface,
  keypair: Keypair,
  subnet_id: int,
  data,
  args: Optional[Any] = None,
):
  """
  Submit consensus data on each epoch with no conditionals

  It is up to prior functions to decide whether to call this function

  :param substrate: interface to blockchain
  :param keypair: keypair of extrinsic caller. Must be a subnet_node in the subnet
  :param consensus_data: an array of data containing all AccountIds, PeerIds, and scores per subnet hoster

  Note: It's important before calling this to ensure the entrinsic will be successful.
        If the function reverts, the extrinsic is Pays::Yes
  """
  # compose call
  call = substrate.compose_call(
    call_module='Network',
    call_function='validate',
    call_params={
      'subnet_id': subnet_id,
      'data': data,
      'args': args,
    }
  )

  # @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(4), after=increment_counter)
  @retry(wait=wait_fixed(BLOCK_SECS+1), stop=stop_after_attempt(4), after=increment_counter)
  def submit_extrinsic():
    try:
      with substrate as _substrate:
        # get none on retries
        nonce = _substrate.get_account_nonce(keypair.ss58_address)

        # create signed extrinsic
        extrinsic = _substrate.create_signed_extrinsic(call=call, keypair=keypair, nonce=nonce)

        receipt = _substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True)
        if receipt.is_success:
          print('✅ Success, triggered events:')
          for event in receipt.triggered_events:
              print(f'* {event.value}')
        else:
            print('⚠️ Extrinsic Failed: ', receipt.error_message)

        return receipt
    except SubstrateRequestException as e:
      print("Failed to send: {}".format(e))

  return submit_extrinsic()

def attest(
  substrate: SubstrateInterface,
  keypair: Keypair,
  subnet_id: int
):
  """
  Submit consensus data on each epoch with no conditionals

  It is up to prior functions to decide whether to call this function

  :param substrate: interface to blockchain
  :param keypair: keypair of extrinsic caller. Must be a subnet_node in the subnet
  :param consensus_data: an array of data containing all AccountIds, PeerIds, and scores per subnet hoster

  Note: It's important before calling this to ensure the entrinsic will be successful.
        If the function reverts, the extrinsic is Pays::Yes
  """
  # compose call
  call = substrate.compose_call(
    call_module='Network',
    call_function='attest',
    call_params={
      'subnet_id': subnet_id,
    }
  )

  # @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(4), after=increment_counter)
  @retry(wait=wait_fixed(BLOCK_SECS+1), stop=stop_after_attempt(4), after=increment_counter)
  def submit_extrinsic():
    try:
      with substrate as _substrate:
        # get none on retries
        nonce = _substrate.get_account_nonce(keypair.ss58_address)

        # create signed extrinsic
        extrinsic = _substrate.create_signed_extrinsic(call=call, keypair=keypair, nonce=nonce)

        receipt = _substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True)

        if receipt.is_success:
          print('✅ Success, triggered events:')
          for event in receipt.triggered_events:
              print(f'* {event.value}')
        else:
            print('⚠️ Extrinsic Failed: ', receipt.error_message)

        return receipt
    except SubstrateRequestException as e:
      print("Failed to send: {}".format(e))

  return submit_extrinsic()

def register_subnet(
  substrate: SubstrateInterface,
  keypair: Keypair,
  path: str,
  memory_mb: int,
  registration_blocks: int
) -> ExtrinsicReceipt:
  """
  Add subnet validator as subnet subnet_node to blockchain storage

  :param substrate: interface to blockchain
  :param keypair: keypair of extrinsic caller. Must be a subnet_node in the subnet
  """

  # compose call
  call = substrate.compose_call(
    call_module='Network',
    call_function='register_subnet',
    call_params={
      'subnet_data': {
        'path': path,
        'memory_mb': memory_mb,
        'registration_blocks': registration_blocks,
      }
    }
  )

  # create signed extrinsic
  extrinsic = substrate.create_signed_extrinsic(call=call, keypair=keypair)

  @retry(wait=wait_fixed(BLOCK_SECS+1), stop=stop_after_attempt(4))
  def submit_extrinsic():
    try:
      with substrate as _substrate:
        receipt = _substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True)
        return receipt
    except SubstrateRequestException as e:
      print("Failed to send: {}".format(e))

  return submit_extrinsic()

def activate_subnet(
  substrate: SubstrateInterface,
  keypair: Keypair,
  subnet_id: str,
) -> ExtrinsicReceipt:
  """
  Add subnet validator as subnet subnet_node to blockchain storage

  :param substrate: interface to blockchain
  :param keypair: keypair of extrinsic caller. Must be a subnet_node in the subnet
  """

  # compose call
  call = substrate.compose_call(
    call_module='Network',
    call_function='activate_subnet',
    call_params={
      'subnet_id': subnet_id,
    }
  )

  # @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(4))
  @retry(wait=wait_fixed(BLOCK_SECS+1), stop=stop_after_attempt(4))
  def submit_extrinsic():
    try:
      with substrate as _substrate:
        # get none on retries
        nonce = _substrate.get_account_nonce(keypair.ss58_address)

        # create signed extrinsic
        extrinsic = _substrate.create_signed_extrinsic(call=call, keypair=keypair, nonce=nonce)

        receipt = _substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True)
        return receipt
    except SubstrateRequestException as e:
      print("Failed to send: {}".format(e))

  return submit_extrinsic()

def remove_subnet(
  substrate: SubstrateInterface,
  keypair: Keypair,
  subnet_id: str,
) -> ExtrinsicReceipt:
  """
  Add subnet validator as subnet subnet_node to blockchain storage

  :param substrate: interface to blockchain
  :param keypair: keypair of extrinsic caller. Must be a subnet_node in the subnet
  """

  # compose call
  call = substrate.compose_call(
    call_module='Network',
    call_function='remove_subnet',
    call_params={
      'subnet_id': subnet_id,
    }
  )

  # create signed extrinsic
  extrinsic = substrate.create_signed_extrinsic(call=call, keypair=keypair)

  @retry(wait=wait_fixed(BLOCK_SECS+1), stop=stop_after_attempt(4))
  def submit_extrinsic():
    try:
      with substrate as _substrate:
        receipt = _substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True)
        return receipt
    except SubstrateRequestException as e:
      print("Failed to send: {}".format(e))

  return submit_extrinsic()

def get_subnet_nodes(
  substrate: SubstrateInterface,
  subnet_id: int,
):
  """
  Function to return all account_ids and subnet_node_ids from the substrate Hypertensor Blockchain

  :param SubstrateInterface: substrate interface from blockchain url
  :returns: subnet_nodes_data
  """
  @retry(wait=wait_fixed(BLOCK_SECS+1), stop=stop_after_attempt(4))
  def make_rpc_request():
    try:
      with substrate as _substrate:
        subnet_nodes_data = _substrate.rpc_request(
          method='network_getSubnetNodes',
          params=[
            subnet_id
          ]
        )
        return subnet_nodes_data
    except SubstrateRequestException as e:
      print("Failed to get rpc request: {}".format(e))

  return make_rpc_request()

def get_subnet_nodes_included(
  substrate: SubstrateInterface,
  subnet_id: int,
):
  """
  Function to return all account_ids and subnet_node_ids from the substrate Hypertensor Blockchain

  :param SubstrateInterface: substrate interface from blockchain url
  :returns: subnet_nodes_data
  """
  @retry(wait=wait_fixed(BLOCK_SECS+1), stop=stop_after_attempt(4))
  def make_rpc_request():
    try:
      with substrate as _substrate:
        subnet_nodes_data = _substrate.rpc_request(
          method='network_getSubnetNodesIncluded',
          params=[subnet_id]
        )
      return subnet_nodes_data
    except SubstrateRequestException as e:
      print("Failed to get rpc request: {}".format(e))

  return make_rpc_request()

def get_subnet_nodes_submittable(
  substrate: SubstrateInterface,
  subnet_id: int,
):
  """
  Function to return all account_ids and subnet_node_ids from the substrate Hypertensor Blockchain

  :param SubstrateInterface: substrate interface from blockchain url
  :returns: subnet_nodes_data
  """
  @retry(wait=wait_fixed(BLOCK_SECS+1), stop=stop_after_attempt(4))
  def make_rpc_request():
    try:
      with substrate as _substrate:
        subnet_nodes_data = _substrate.rpc_request(
          method='network_getSubnetNodesSubmittable',
          params=[
            subnet_id
          ]
        )
        return subnet_nodes_data
    except SubstrateRequestException as e:
      print("Failed to get rpc request: {}".format(e))

  return make_rpc_request()

async def get_consensus_data(
  substrate: SubstrateInterface,
  subnet_id: int,
  epoch: int
):
  """
  Function to return all account_ids and subnet_node_ids from the substrate Hypertensor Blockchain

  :param SubstrateInterface: substrate interface from blockchain url
  :returns: subnet_nodes_data
  """
  @retry(wait=wait_fixed(BLOCK_SECS+1), stop=stop_after_attempt(4))
  def make_rpc_request():
    try:
      with substrate as _substrate:
        subnet_nodes_data = _substrate.rpc_request(
          method='network_getConsensusData',
          params=[
            subnet_id,
            epoch
          ]
        )
        return subnet_nodes_data
    except SubstrateRequestException as e:
      print("Failed to get rpc request: {}".format(e))

  return make_rpc_request()

async def get_accountant_data(
  substrate: SubstrateInterface,
  subnet_id: int,
  id: int
):
  """
  Function to return all account_ids and subnet_node_ids from the substrate Hypertensor Blockchain

  :param SubstrateInterface: substrate interface from blockchain url
  :returns: subnet_nodes_data
  """
  @retry(wait=wait_fixed(BLOCK_SECS+1), stop=stop_after_attempt(4))
  def make_rpc_request():
    try:
      with substrate as _substrate:
        subnet_nodes_data = _substrate.rpc_request(
          method='network_getAccountantData',
          params=[
            subnet_id,
            id
          ]
        )
        return subnet_nodes_data
    except SubstrateRequestException as e:
      print("Failed to get rpc request: {}".format(e))

  return make_rpc_request()

def is_subnet_node_by_peer_id(
  substrate: SubstrateInterface,
  subnet_id: int,
  peer_id: str
):
  """
  Function to return all account_ids and subnet_node_ids from the substrate Hypertensor Blockchain

  :param SubstrateInterface: substrate interface from blockchain url
  :returns: subnet_nodes_data
  """
  @retry(wait=wait_fixed(BLOCK_SECS+1), stop=stop_after_attempt(4))
  def make_rpc_request():
    try:
      with substrate as _substrate:
        is_subnet_node = _substrate.rpc_request(
          method='network_isSubnetNodeByPeerId',
          params=[
            subnet_id,
            peer_id
          ]
        )
        return is_subnet_node
    except SubstrateRequestException as e:
      print("Failed to get rpc request: {}".format(e))

  return make_rpc_request()

def get_minimum_subnet_nodes(
  substrate: SubstrateInterface,
  memory_mb: int,
):
  """
  Function to return all account_ids and subnet_node_ids from the substrate Hypertensor Blockchain

  :param SubstrateInterface: substrate interface from blockchain url
  :returns: subnet_nodes_data
  """
  @retry(wait=wait_fixed(BLOCK_SECS+1), stop=stop_after_attempt(4))
  def make_rpc_request():
    try:
      with substrate as _substrate:
        subnet_nodes_data = _substrate.rpc_request(
          method='network_getMinimumSubnetNodes',
          params=[
            memory_mb
          ]
        )
        return subnet_nodes_data
    except SubstrateRequestException as e:
      print("Failed to get rpc request: {}".format(e))

  return make_rpc_request()

def get_minimum_delegate_stake(
  substrate: SubstrateInterface,
  memory_mb: int,
):
  """
  Function to return all account_ids and subnet_node_ids from the substrate Hypertensor Blockchain

  :param SubstrateInterface: substrate interface from blockchain url
  :returns: subnet_nodes_data
  """
  @retry(wait=wait_fixed(BLOCK_SECS+1), stop=stop_after_attempt(4))
  def make_rpc_request():
    try:
      with substrate as _substrate:
        subnet_nodes_data = _substrate.rpc_request(
          method='network_getMinimumDelegateStake',
          params=[
            memory_mb
          ]
        )
        return subnet_nodes_data
    except SubstrateRequestException as e:
      print("Failed to get rpc request: {}".format(e))

  return make_rpc_request()

def get_subnet_node_info(
  substrate: SubstrateInterface,
  subnet_id: int,
):
  """
  Function to return all account_ids and subnet_node_ids from the substrate Hypertensor Blockchain

  :param SubstrateInterface: substrate interface from blockchain url
  :returns: subnet_nodes_data
  """
  @retry(wait=wait_fixed(BLOCK_SECS+1), stop=stop_after_attempt(4))
  def make_rpc_request():
    try:
      with substrate as _substrate:
        subnet_nodes_data = _substrate.rpc_request(
          method='network_getSubnetNodeInfo',
          params=[
            subnet_id
          ]
        )
        return subnet_nodes_data
    except SubstrateRequestException as e:
      print("Failed to get rpc request: {}".format(e))

  return make_rpc_request()

def add_subnet_node(
  substrate: SubstrateInterface,
  keypair: Keypair,
  subnet_id: int,
  hotkey: str,
  peer_id: str,
  stake_to_be_added: int,
  a: Optional[str] = None,
  b: Optional[str] = None,
  c: Optional[str] = None,
) -> ExtrinsicReceipt:
  """
  Add subnet validator as subnet subnet_node to blockchain storage

  :param substrate: interface to blockchain
  :param keypair: keypair of extrinsic caller. Must be a subnet_node in the subnet
  """

  # compose call
  call = substrate.compose_call(
    call_module='Network',
    call_function='add_subnet_node',
    call_params={
      'subnet_id': subnet_id,
      'hotkey': hotkey,
      'peer_id': peer_id,
      'stake_to_be_added': stake_to_be_added,
      'a': a,
      'b': b,
      'c': c,
    }
  )

  @retry(wait=wait_fixed(BLOCK_SECS+1), stop=stop_after_attempt(4))
  def submit_extrinsic():
    try:
      with substrate as _substrate:
        # get none on retries
        nonce = _substrate.get_account_nonce(keypair.ss58_address)

        # create signed extrinsic
        extrinsic = _substrate.create_signed_extrinsic(call=call, keypair=keypair, nonce=nonce)

        receipt = _substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True)
        return receipt
    except SubstrateRequestException as e:
      print("Failed to send: {}".format(e))

  return submit_extrinsic()

def register_subnet_node(
  substrate: SubstrateInterface,
  keypair: Keypair,
  subnet_id: int,
  hotkey: str,
  peer_id: str,
  stake_to_be_added: int,
  a: Optional[str] = None,
  b: Optional[str] = None,
  c: Optional[str] = None,
) -> ExtrinsicReceipt:
  """
  Add subnet validator as subnet subnet_node to blockchain storage

  :param substrate: interface to blockchain
  :param keypair: keypair of extrinsic caller. Must be a subnet_node in the subnet
  """

  # compose call
  call = substrate.compose_call(
    call_module='Network',
    call_function='register_subnet_node',
    call_params={
      'subnet_id': subnet_id,
      'hotkey': hotkey,
      'peer_id': peer_id,
      'stake_to_be_added': stake_to_be_added,
      'a': a,
      'b': b,
      'c': c,
    }
  )

  @retry(wait=wait_fixed(BLOCK_SECS+1), stop=stop_after_attempt(4))
  def submit_extrinsic():
    try:
      with substrate as _substrate:
        # get none on retries
        nonce = _substrate.get_account_nonce(keypair.ss58_address)

        # create signed extrinsic
        extrinsic = _substrate.create_signed_extrinsic(call=call, keypair=keypair, nonce=nonce)

        receipt = _substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True)
        return receipt
    except SubstrateRequestException as e:
      print("Failed to send: {}".format(e))

  return submit_extrinsic()

def register_subnet_node_v1(
  substrate: SubstrateInterface,
  keypair: Keypair,
  subnet_id: int,
  peer_id: str,
  stake_to_be_added: int,
  a: Optional[str] = None,
  b: Optional[str] = None,
  c: Optional[str] = None,
) -> ExtrinsicReceipt:
  """
  Add subnet validator as subnet subnet_node to blockchain storage

  :param substrate: interface to blockchain
  :param keypair: keypair of extrinsic caller. Must be a subnet_node in the subnet
  """

  # compose call
  call = substrate.compose_call(
    call_module='Network',
    call_function='register_subnet_node',
    call_params={
      'subnet_id': subnet_id,
      'peer_id': peer_id,
      'stake_to_be_added': stake_to_be_added,
      'a': a,
      'b': b,
      'c': c,
    }
  )

  @retry(wait=wait_fixed(BLOCK_SECS+1), stop=stop_after_attempt(4))
  def submit_extrinsic():
    try:
      with substrate as _substrate:
        # get none on retries
        nonce = _substrate.get_account_nonce(keypair.ss58_address)

        # create signed extrinsic
        extrinsic = _substrate.create_signed_extrinsic(call=call, keypair=keypair, nonce=nonce)

        receipt = _substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True)
        return receipt
    except SubstrateRequestException as e:
      print("Failed to send: {}".format(e))

  return submit_extrinsic()

def activate_subnet_node(
  substrate: SubstrateInterface,
  keypair: Keypair,
  subnet_id: int,
) -> ExtrinsicReceipt:
  """
  Add subnet validator as subnet subnet_node to blockchain storage

  :param substrate: interface to blockchain
  :param keypair: keypair of extrinsic caller. Must be a subnet_node in the subnet
  """

  # compose call
  call = substrate.compose_call(
    call_module='Network',
    call_function='activate_subnet_node',
    call_params={
      'subnet_id': subnet_id,
    }
  )

  @retry(wait=wait_fixed(BLOCK_SECS+1), stop=stop_after_attempt(4))
  def submit_extrinsic():
    try:
      with substrate as _substrate:
        # get none on retries
        nonce = _substrate.get_account_nonce(keypair.ss58_address)

        # create signed extrinsic
        extrinsic = _substrate.create_signed_extrinsic(call=call, keypair=keypair, nonce=nonce)

        receipt = _substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True)
        return receipt
    except SubstrateRequestException as e:
      print("Failed to send: {}".format(e))

  return submit_extrinsic()

def deactivate_subnet_node(
  substrate: SubstrateInterface,
  keypair: Keypair,
  subnet_id: int,
) -> ExtrinsicReceipt:
  """
  Add subnet validator as subnet subnet_node to blockchain storage

  :param substrate: interface to blockchain
  :param keypair: keypair of extrinsic caller. Must be a subnet_node in the subnet
  """

  # compose call
  call = substrate.compose_call(
    call_module='Network',
    call_function='activate_subnet_node',
    call_params={
      'subnet_id': subnet_id,
    }
  )

  @retry(wait=wait_fixed(BLOCK_SECS+1), stop=stop_after_attempt(4))
  def submit_extrinsic():
    try:
      with substrate as _substrate:
        # get none on retries
        nonce = _substrate.get_account_nonce(keypair.ss58_address)

        # create signed extrinsic
        extrinsic = _substrate.create_signed_extrinsic(call=call, keypair=keypair, nonce=nonce)

        receipt = _substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True)
        return receipt
    except SubstrateRequestException as e:
      print("Failed to send: {}".format(e))

  return submit_extrinsic()

def remove_subnet_node(
  substrate: SubstrateInterface,
  keypair: Keypair,
  subnet_id: int,
):
  """
  Remove stake balance towards specified subnet

  Amount must be less than allowed amount that won't allow stake balance to be lower than
  the required minimum balance

  :param substrate: interface to blockchain
  :param keypair: keypair of extrinsic caller. Must be a subnet_node in the subnet
  """

  # compose call
  call = substrate.compose_call(
    call_module='Network',
    call_function='remove_subnet_node',
    call_params={
      'subnet_id': subnet_id,
    }
  )

  @retry(wait=wait_fixed(BLOCK_SECS+1), stop=stop_after_attempt(4))
  def submit_extrinsic():
    try:
      with substrate as _substrate:
        # get none on retries
        nonce = _substrate.get_account_nonce(keypair.ss58_address)

        # create signed extrinsic
        extrinsic = _substrate.create_signed_extrinsic(call=call, keypair=keypair, nonce=nonce)

        receipt = _substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True)
        return receipt
    except SubstrateRequestException as e:
      print("Failed to send: {}".format(e))

  return submit_extrinsic()

def add_to_stake(
  substrate: SubstrateInterface,
  keypair: Keypair,
  subnet_id: int,
  stake_to_be_added: int,
):
  """
  Add subnet validator as subnet subnet_node to blockchain storage

  :param substrate: interface to blockchain
  :param keypair: keypair of extrinsic caller. Must be a subnet_node in the subnet
  :param stake_to_be_added: stake to be added towards subnet
  """

  # compose call
  call = substrate.compose_call(
    call_module='Network',
    call_function='add_to_stake',
    call_params={
      'subnet_id': subnet_id,
      'stake_to_be_added': stake_to_be_added,
    }
  )

  @retry(wait=wait_fixed(BLOCK_SECS+1), stop=stop_after_attempt(4))
  def submit_extrinsic():
    try:
      with substrate as _substrate:
        # get none on retries
        nonce = _substrate.get_account_nonce(keypair.ss58_address)

        # create signed extrinsic
        extrinsic = _substrate.create_signed_extrinsic(call=call, keypair=keypair, nonce=nonce)

        receipt = _substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True)
        return receipt
    except SubstrateRequestException as e:
      print("Failed to send: {}".format(e))

  return submit_extrinsic()

def remove_stake(
  substrate: SubstrateInterface,
  keypair: Keypair,
  subnet_id: int,
  stake_to_be_removed: int,
):
  """
  Remove stake balance towards specified subnet

  Amount must be less than allowed amount that won't allow stake balance to be lower than
  the required minimum balance

  :param substrate: interface to blockchain
  :param keypair: keypair of extrinsic caller. Must be a subnet_node in the subnet
  :param stake_to_be_removed: stake to be removed from subnet
  """

  # compose call
  call = substrate.compose_call(
    call_module='Network',
    call_function='remove_stake',
    call_params={
      'subnet_id': subnet_id,
      'stake_to_be_removed': stake_to_be_removed,
    }
  )

  @retry(wait=wait_fixed(BLOCK_SECS+1), stop=stop_after_attempt(4))
  def submit_extrinsic():
    try:
      with substrate as _substrate:
        # get none on retries
        nonce = _substrate.get_account_nonce(keypair.ss58_address)

        # create signed extrinsic
        extrinsic = _substrate.create_signed_extrinsic(call=call, keypair=keypair, nonce=nonce)

        receipt = _substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True)
        return receipt
    except SubstrateRequestException as e:
      print("Failed to send: {}".format(e))

  return submit_extrinsic()

def add_to_delegate_stake(
  substrate: SubstrateInterface,
  keypair: Keypair,
  subnet_id: int,
  stake_to_be_added: int,
):
  """
  Add subnet validator as subnet subnet_node to blockchain storage

  :param substrate: interface to blockchain
  :param keypair: keypair of extrinsic caller. Must be a subnet_node in the subnet
  :param stake_to_be_added: stake to be added towards subnet
  """

  # compose call
  call = substrate.compose_call(
    call_module='Network',
    call_function='add_to_delegate_stake',
    call_params={
      'subnet_id': subnet_id,
      'stake_to_be_added': stake_to_be_added,
    }
  )

  @retry(wait=wait_fixed(BLOCK_SECS+1), stop=stop_after_attempt(4))
  def submit_extrinsic():
    try:
      with substrate as _substrate:
        # get none on retries
        nonce = _substrate.get_account_nonce(keypair.ss58_address)

        # create signed extrinsic
        extrinsic = _substrate.create_signed_extrinsic(call=call, keypair=keypair, nonce=nonce)

        receipt = _substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True)
        return receipt
    except SubstrateRequestException as e:
      print("Failed to send: {}".format(e))

  return submit_extrinsic()

def get_balance(
  substrate: SubstrateInterface,
  address: str
):
  """
  Function to return account balance

  :param SubstrateInterface: substrate interface from blockchain url
  :param address: address of account_id
  :returns: account balance
  """
  @retry(wait=wait_fixed(BLOCK_SECS+1), stop=stop_after_attempt(4))
  def make_query():
    try:
      with substrate as _substrate:
        result = _substrate.query('System', 'Account', [address])
        return result.value['data']['free']
    except SubstrateRequestException as e:
      print("Failed to get rpc request: {}".format(e))

  return make_query()

def get_subnet_stake_balance(
  substrate: SubstrateInterface,
  subnet_id: int,
  address: str
):
  """
  Function to return an accounts stake balance towards a subnet

  :param SubstrateInterface: substrate interface from blockchain url
  :param address: address of account_id
  :returns: account stake balance towards subnet
  """
  @retry(wait=wait_fixed(BLOCK_SECS+1), stop=stop_after_attempt(4))
  def make_query():
    try:
      with substrate as _substrate:
        result = _substrate.query('Network', 'AccountSubnetStake', [address, subnet_id])
        return result
    except SubstrateRequestException as e:
      print("Failed to get rpc request: {}".format(e))

  return make_query()

def get_subnet_node_account(
  substrate: SubstrateInterface,
  subnet_id: int,
  peer_id: str
):
  """
  Function to account_id of subnet hosting subnet_node

  :param SubstrateInterface: substrate interface from blockchain url
  :param peer_id: peer_id of subnet validator
  :returns: account_id of subnet_id => peer_id
  """
  @retry(wait=wait_fixed(BLOCK_SECS+1), stop=stop_after_attempt(4))
  def make_query():
    try:
      with substrate as _substrate:
        result = _substrate.query('Network', 'SubnetNodeAccount', [subnet_id, peer_id])
        return result
    except SubstrateRequestException as e:
      print("Failed to get rpc request: {}".format(e))

  return make_query()

def get_subnet_accounts(
  substrate: SubstrateInterface,
  subnet_id: int,
):
  """
  Function to account_id of subnet hosting subnet_node

  :param SubstrateInterface: substrate interface from blockchain url
  :returns: account_id's of subnet_id
  """
  @retry(wait=wait_fixed(BLOCK_SECS+1), stop=stop_after_attempt(4))
  def make_query():
    try:
      with substrate as _substrate:
        result = _substrate.query('Network', 'SubnetAccount', [subnet_id])
        return result
    except SubstrateRequestException as e:
      print("Failed to get rpc request: {}".format(e))

  return make_query()

def get_subnet_id_by_path(
  substrate: SubstrateInterface,
  path: str
):
  """
  Function to get python -m subnet.cli.crypto.keygen --path private_key3.key
 of subnet hosting subnet_node

  :param SubstrateInterface: substrate interface from blockchain url
  :param path: path of subnet
  :returns: subnet_id
  """
  @retry(wait=wait_fixed(BLOCK_SECS+1), stop=stop_after_attempt(4))
  def make_query():
    try:
      with substrate as _substrate:
        result = _substrate.query('Network', 'SubnetPaths', [path])
        return result
    except SubstrateRequestException as e:
      print("Failed to get rpc request: {}".format(e))

  return make_query()

def get_subnet_data(
  substrate: SubstrateInterface,
  id: int
):
  """
  Function to get data struct of the subnet

  :param SubstrateInterface: substrate interface from blockchain url
  :param id: id of subnet
  :returns: subnet_id
  """
  @retry(wait=wait_fixed(BLOCK_SECS+1), stop=stop_after_attempt(4))
  def make_query():
    try:
      with substrate as _substrate:
        result = _substrate.query('Network', 'SubnetsData', [id])
        return result
    except SubstrateRequestException as e:
      print("Failed to get rpc request: {}".format(e))

  return make_query()

def get_max_subnets(substrate: SubstrateInterface):
  """
  Function to get the maximum number of subnets allowed on the blockchain

  :param SubstrateInterface: substrate interface from blockchain url
  :returns: max_subnets
  """

  @retry(wait=wait_fixed(BLOCK_SECS+1), stop=stop_after_attempt(4))
  def make_query():
    try:
      with substrate as _substrate:
        result = _substrate.query('Network', 'MaxSubnets')
        return result
    except SubstrateRequestException as e:
      print("Failed to get rpc request: {}".format(e))

  return make_query()

def get_min_subnet_nodes(substrate: SubstrateInterface):
  """
  Function to get the minimum number of subnet_nodes required to host a subnet

  :param SubstrateInterface: substrate interface from blockchain url
  :returns: min_subnet_nodes
  """

  @retry(wait=wait_fixed(BLOCK_SECS+1), stop=stop_after_attempt(4))
  def make_query():
    try:
      with substrate as _substrate:
        result = _substrate.query('Network', 'MinSubnetNodes')
        return result
    except SubstrateRequestException as e:
      print("Failed to get rpc request: {}".format(e))

  return make_query()

def get_min_stake_balance(substrate: SubstrateInterface):
  """
  Function to get the minimum stake balance required to host a subnet
  
  :param SubstrateInterface: substrate interface from blockchain url
  :returns: min_stake_balance
  """

  @retry(wait=wait_fixed(BLOCK_SECS+1), stop=stop_after_attempt(4))
  def make_query():
    try:
      with substrate as _substrate:
        result = _substrate.query('Network', 'MinStakeBalance')
        return result
    except SubstrateRequestException as e:
      print("Failed to get rpc request: {}".format(e))

  return make_query()

def get_max_subnet_nodes(substrate: SubstrateInterface):
  """
  Function to get the maximum number of subnet_nodes allowed to host a subnet

  :param SubstrateInterface: substrate interface from blockchain url
  :returns: max_subnet_nodes
  """

  @retry(wait=wait_fixed(BLOCK_SECS+1), stop=stop_after_attempt(4))
  def make_query():
    try:
      with substrate as _substrate:
        result = _substrate.query('Network', 'MaxSubnetNodes')
        return result
    except SubstrateRequestException as e:
      print("Failed to get rpc request: {}".format(e))

  return make_query()

def get_tx_rate_limit(substrate: SubstrateInterface):
  """
  Function to get the transaction rate limit

  :param SubstrateInterface: substrate interface from blockchain url
  :returns: tx_rate_limit
  """
  
  @retry(wait=wait_fixed(BLOCK_SECS+1), stop=stop_after_attempt(4))
  def make_query():
    try:
      with substrate as _substrate:
        result = _substrate.query('Network', 'TxRateLimit')
        return result
    except SubstrateRequestException as e:
      print("Failed to get rpc request: {}".format(e))

  return make_query()

def get_epoch_length(substrate: SubstrateInterface):
  """
  Function to get the epoch length as blocks per epoch

  :param SubstrateInterface: substrate interface from blockchain url
  :returns: epoch_length
  """

  @retry(wait=wait_fixed(BLOCK_SECS+1), stop=stop_after_attempt(4))
  def make_query():
    try:
      with substrate as _substrate:
        result = _substrate.get_constant('Network', 'EpochLength')
        return result
    except SubstrateRequestException as e:
      print("Failed to get rpc request: {}".format(e))

  return make_query()

def get_rewards_validator(
  substrate: SubstrateInterface,
  subnet_id: int,
  epoch: int
):
  """
  Function to get the consensus blocks interval

  :param SubstrateInterface: substrate interface from blockchain url
  :returns: epoch_length
  """

  @retry(wait=wait_fixed(BLOCK_SECS+1), stop=stop_after_attempt(4))
  def make_query():
    try:
      with substrate as _substrate:
        result = _substrate.query('Network', 'SubnetRewardsValidator', [subnet_id, epoch])
        return result
    except SubstrateRequestException as e:
      print("Failed to get rpc request: {}".format(e))

  return make_query()

def get_rewards_submission(
  substrate: SubstrateInterface,
  subnet_id: int,
  epoch: int
):
  """
  Function to get the consensus blocks interval

  :param SubstrateInterface: substrate interface from blockchain url
  :returns: epoch_length
  """

  @retry(wait=wait_fixed(BLOCK_SECS+1), stop=stop_after_attempt(4))
  def make_query():
    try:
      with substrate as _substrate:
        result = _substrate.query('Network', 'SubnetRewardsSubmission', [subnet_id, epoch])
        return result
    except SubstrateRequestException as e:
      print("Failed to get rpc request: {}".format(e))

  return make_query()

def get_min_subnet_registration_blocks(substrate: SubstrateInterface):
  """
  Function to get the consensus blocks interval

  :param SubstrateInterface: substrate interface from blockchain url
  :returns: epoch_length
  """

  @retry(wait=wait_fixed(BLOCK_SECS+1), stop=stop_after_attempt(4))
  def make_query():
    try:
      with substrate as _substrate:
        result = _substrate.query('Network', 'MinSubnetRegistrationBlocks')
        return result
    except SubstrateRequestException as e:
      print("Failed to get rpc request: {}".format(e))

  return make_query()

def get_max_subnet_registration_blocks(substrate: SubstrateInterface):
  """
  Function to get the consensus blocks interval

  :param SubstrateInterface: substrate interface from blockchain url
  :returns: epoch_length
  """

  @retry(wait=wait_fixed(BLOCK_SECS+1), stop=stop_after_attempt(4))
  def make_query():
    try:
      with substrate as _substrate:
        result = _substrate.query('Network', 'MaxSubnetRegistrationBlocks')
        return result
    except SubstrateRequestException as e:
      print("Failed to get rpc request: {}".format(e))

  return make_query()



# EVENTS

def get_reward_result_event(
  substrate: SubstrateInterface,
  target_subnet_id: int,
  epoch: int
):
  """
  Function to return all account_ids and subnet_node_ids from the substrate Hypertensor Blockchain

  :param SubstrateInterface: substrate interface from blockchain url
  :returns: subnet_nodes_data
  """

  @retry(wait=wait_fixed(BLOCK_SECS+1), stop=stop_after_attempt(4))
  def make_event_query():
    try:
      epoch_length = get_epoch_length(substrate)
      epoch_length = int(str(epoch_length))
      block_number = epoch_length * epoch
      block_hash = substrate.get_block_hash(block_number=block_number)
      with substrate as _substrate:
        data = None
        events = _substrate.get_events(block_hash=block_hash)
        for event in events:
          if event['event']['module_id'] == "Network" and event['event']['event_id'] == "RewardResult":
            subnet_id, attestation_percentage = event['event']['attributes']
            if subnet_id == target_subnet_id:
              data = subnet_id, attestation_percentage
              break
        return data
    except SubstrateRequestException as e:
      print("Failed to get rpc request: {}".format(e))

  return make_event_query()
