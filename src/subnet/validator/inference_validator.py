import itertools
import random
import threading
from typing import Dict, List, Optional
import dataclasses
import time
import multiprocessing as mp

from subnet.validator.config import AccountantData, AccountantDataPeerParams, PeerInferenceResults, PeerInferenceSequenceData, PeerValidationData
# from subnet.validator.routing.sequence_manager import MissingBlocksError
# from subnet.data_structures import RemoteSpanInfo
from subnet.health.state_updater import get_peers_data_list
# from subnet.substrate.chain_functions import propose_model_peer_dishonest, vote_model_peer_dishonest
from subnet.utils.auto_config import AutoDistributedModelForCausalLMValidator
# from subnet.substrate import config as substrate_config

from transformers import AutoTokenizer
import torch
from hypermind.utils.logging import get_logger
from hypermind import PeerID, DHT
from hypermind.utils.auth import AuthorizerBase

import pprint 

logger = get_logger(__name__)

# TODO: make substrate_config a class
# from hypertensor import HypertensorClient

"""Timespan per inference validation per peer"""
TIMESPAN_PER_PEER = 300
STRICT_BLOCK_WAIT = 300
MODULE_CONTAINER_WAIT = 90
# seconds between periodically validating node
POI_PERIOD = 86400              # 24 hours

"""Inference configuration"""

# DO NOT change these. All accountants must have the same inference validation configuration
RTOL = 1e-03 # relative
ATOL = 8e-01 # absolute

VTOL = 0.8 # valid tolerance of each peers position in the sequence, this is a percentage of each positions validity

class InferenceValidator(threading.Thread):
    """
    Runs Inference validation logic, runs per epoch,
    called by Server to start processing before ModuleContainers load, 
    inherits Server reference to update variables in the Server,
    and updates the server's state.
    """

    def __init__(
        self, 
        server, 
        model_name, 
        num_model_blocks: int, 
        dht: DHT,
        num_blocks: int, 
        authorizer: AuthorizerBase,
        identity_path: str,
        start: bool,
    ):
        super().__init__()
        self.server = server # Server()
        self.dht = dht
        self.model = None
        self.my_peer_id = dht.peer_id
        self.model_name = model_name
        self.num_blocks = num_blocks # num blocks node is powering
        self.num_model_blocks = num_model_blocks
        # self.ranges = list(itertools.combinations(range(0,num_model_blocks+1), 2))
        self.authorizer = authorizer
        self.identity_path = identity_path

        #
        # Blockchain information
        #
        self.model_id = None
        self.epoch = 0

        #
        # Validation variables
        #
        self.peers_data = None
        self.peers_data_to_validate = None
        self.cached_inference_sequence = None
        self.input_data = "A cat sat"
        # Simple grammar test inputs
        # TODO: Add environment variables for this so users can update them as they please
        self.inputs = [
            "The cat chases",
            "A dog barked",
        ]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # If peers preceding in the sequence are not valid, everyone after them will be invalid
        # We will need to rerun the sequence and recheck
        self.last_timestamp = 0
        # list of possibly nodes that are dishonest to qualify against others before submitting a proposal
        self.poss_fault_nodes = None

        #
        # Data for blockchain proposals
        #
        self.accountant_data = AccountantData()



        # Is this epochs accountant required to submit data
        self.is_accountant = False



        # TODO: Get blacklisted peers to automatically create a dishonesty proposal for them


        # TODO: Run inference sequence on multiple inputs and store them in a pickle file
        #       Then the inference validator will choose at random each time they are chosen
        #       to be an Accountant. This will limit the computation needed for PoI
        """
        cached_inference_sequences = [
            {
                sequence: List,
                last_used: int
            }
        ]
        """
        self.cached_inference_sequences = None

        if start:
            self.run_in_background(await_ready=True)

        self.stop = threading.Event()

    def run(self):
        """
        Listens for proposals while running inference validation PoI
        Only one should be running at a time.
        When a proposal is found, shut down the inference validator and validate the proposal data
        """
        t1 = threading.Thread(target=self.run_validator(), args=())
        t1.start()
        t1.join()

    def run_validator(self):
        """
            1. Find the least amount of blocks to validate
                - We find the lowest end block and the highest start block as the span to validate
                - Accountant always starts at 0 though instead of the lowest end block until future iterations
            2. Run inference on the least amount of blocks starting at 0 up to the max block using Accountant only for inference sequence
                - This can take multiple iterations to complete
                    - e.g. If the accountant has 40/80 blocks and the max block is 80, it will take
                    2 iterations to complete
                - This sequence data is cached locally to use in validations in order to save on compute
                  for the validating remote sequence runs including peers
                - By using only the Accountant in the sequence, the accountants data is assumed honest so others can not corrupt
                  the sequence data
                    - This also allows other accountants to know for sure that another accountant is dishonest because the data they 
                      submit is always checked against their own
            3. Run inference of other peers to validate by injecting them inside the blocks using the cached data
                - Only the block(s) used to validate peers are used in the sequence for inference
                - The other blocks not selected for the sequence are not ran and the cached data is used in its place instead
            4. Validate accountant inference outputs from the first accountant-only runs versus the peers we validated
                - Validate by checking relative and absolute tolerances
            -  If chosen epoch-Accountant, submit all data of each peers data to the blockchain
            -  If an Accountant in general, and found a dishonest peer, submit a dishonesty proposal


            Servers:        [00] [01] [02] [03] [04] [05] [06] [07] [08] [09] [10] [11] [12]
            Server Indexes: [   0   ]    |    |    |    |    |    |    |    |    |    |    |
                                 [   1   ]    |    |    |    |    |    |    |    |    |    |
                                      [   2   ]    |    |    |    |    |    |    |    |    |
                                           [   3   ]    |    |    |    |    |    |    |    |
                                                [   4   ]    |    |    |    |    |    |    |
                                                     [   5   ]    |    |    |    |    |    |
                                                          [   6   ]    |    |    |    |    |
                                                               [   7   ]    |    |    |    |
                                                                    [   8   ]    |    |    |
                                                                         [   9   ]    |    |
                                                                              [   10  ]    |
                                                                                   [   11  ]

            • Only run if is an accountant, otherwise wait until is accountant

            • If is accountant:
                • If is chosen accountant:
                    • Run validation PoI on all nodes
                    • Submit data to blockchain
                    • If faulty node found:
                        • Submit proposal to remove node from blockchain
                            • If removed, should be automatically removed from routing table
                • If not chosen accountant
                    • Run PoI on newly entered nodes that have not been validated yet
                    • Run PoI on all nodes periodically
        """            
        while True:
            # Begin epoch 
            epoch = self._get_epoch()

            # only run if an accountant
            if not self._is_accountant():
                seconds_remaining_in_epoch = self._get_seconds_remaining_in_epoch()
                logger.info(f"Next epoch is {seconds_remaining_in_epoch} seconds away, sleeping for remaining time")
                time.sleep(seconds_remaining_in_epoch)
                continue
            
            if self._is_chosen_accountant():
                # If chosen accountant, submit all data of each peers data to the blockchain
                ...

            # Reset previous epochs cached inference sequence
            self.cached_inference_sequence = None

            # Reset accountant data for the new epoch
            self.accountant_data.reset()

            # Ensure that module container is created
            if not self._is_module_container_healthy():
                time.sleep(30)
                continue

            try:
                logger.info("Setting status to validator")
                self.server.is_validator = True

                if self.epoch == 0 and (self.peers_data_to_validate is None or len(self.peers_data_to_validate) == 0):
                    # Restart loop
                    logger.info("Updating peers within the epoch")
                    self.update_peers()

                # Ensure other peers are present to validate
                if self.peers_data_to_validate == None or len(self.peers_data_to_validate) == 0:
                    logger.info("There are zero peers, waiting for other peers to join")
                    break

                logger.info("Getting sequences of peers to validate")
                peers_validation_spans = self.get_peers_validation_spans()
                logger.info(f"Found {len(peers_validation_spans)} sequences")

                # Get max start block 
                logger.info(f"Getting min/max blocks for Accountant to get inference data of")
                blocks = self.get_min_max_start_block_from_sequences(peers_validation_spans)

                min_block = 0
                max_block = blocks[1]
                num_blocks = self.num_blocks
                if num_blocks > max_block:
                    num_blocks = max_block

                logger.info("Getting all ranges for Accountant to run inference sequences based on peers distributions")
                spans = []
                for index, value in enumerate(range(0, max_block+max_block-min_block, num_blocks)):
                    if index == 0:
                        spans.append([value, value + num_blocks])
                    else:
                        if value + num_blocks > max_block:
                            if value + num_blocks - num_blocks != max_block:
                                spans.append([value + num_blocks - num_blocks, max_block])
                            break
                        else:
                            spans.append([value + num_blocks - num_blocks, value + num_blocks])

                print("spans", spans)

                logger.info("Gathering sequence to run as Accountant to cache the data")
                accountant_spans = []
                for span in spans:
                    start = span[0]
                    end = span[1]

                    accountant_span_ranges = []
                    for index, value in enumerate(range(start, end+end-start, 2)):
                        if index == 0:
                            accountant_span_ranges.append({
                                'peer_id':self.my_peer_id,
                                'start':value,
                                'end':value + 1,
                            })
                        else:
                            accountant_span_ranges.append({
                                'peer_id':self.my_peer_id,
                                'start':value-index,
                                'end':value-index + 1,
                            })

                    accountant_spans.append(accountant_span_ranges)

                if self.model is None:
                    self.model = AutoDistributedModelForCausalLMValidator.from_pretrained(self.model_name, identity_path="private_key2.key")

                logger.info("Running inference on accountant spans and storing inference results")
                for accountant_span in accountant_spans:
                    span_start = accountant_span[0]["start"]
                    span_end = accountant_span[-1]["end"]

                    block_indices = f"{span_start}:{span_end}"
                    print("accountant_span block_indices", block_indices)

                    logger.info(f"Updating strict blocks to {block_indices} if needed")
                    self.server.update_strict_block_indices(block_indices)
                    while True:
                        # Ensure that module container is created
                        if not self._is_module_container_healthy():
                            time.sleep(MODULE_CONTAINER_WAIT)
                            continue

                        start_block = self.server.module_container.server_info.start_block
                        end_block = self.server.module_container.server_info.end_block

                        if start_block != span_start and end_block != span_end:
                            logger.info("Blocks don't match yet")
                            time.sleep(MODULE_CONTAINER_WAIT)
                            continue

                        # Example: Wait until blocks are updated
                        # This is a hack attempt - need to instead check for that blocks have been updated to the correct spans
                        if not self._is_module_container_healthy():
                            time.sleep(MODULE_CONTAINER_WAIT)
                            continue
                        
                        logger.info(f"Running inference as an Accountant on block indices {block_indices} and storing results")
                        self.run_inference_as_accountant(
                            self.input_data, 
                            peers=accountant_span
                        )

                        # once successful, break the loop
                        break

                logger.info("Complete inference sequence as Accountant using self")
                logger.info(f"Accountant has {len(self.cached_inference_sequence)} results cached")

                ####
                # Go peer by peer using cached data and injecting peer inside sequence to limit computations
                # Current implementation only supports double spans as in 0:1 or 2:3
                logger.info(f"Building sequence of spans {block_indices} to run sequence with other peers")                
                for peer_validation_span in peers_validation_spans:
                    # Iterate and validate peer by peer
                    for peer in peer_validation_span:
                        cached_server_sessions = []
                        span_ranges = []
                        block = 0
                        # Get full range of cache history if available to limit computation
                        while block <= max_block:
                            if peer["start"] == block:
                                # Add peer to sequence
                                span_ranges.append(peer)
                                block = peer["end"]
                            else:
                                # Add accountant sequence cache to sequence
                                input_tensors = self.get_account_input_tensors(block, block+1)
                                print("\npeer_validation_span input_tensors")
                                print("\n peer validation span", block, block+1)
                                pprint.pprint(input_tensors)
                                if input_tensors is not None:
                                    cached_server_sessions.append(input_tensors)
                                    block = input_tensors[0]["span_end"]
                                else:
                                    # If None, it will be filled in automatically when running the remote inference sequence
                                    block += 1

                        # Combine cached_server_sessions into one array
                        sequence_tensors = []
                        for arr in cached_server_sessions:
                            sequence_tensors += arr

                        logger.info(f"Running inference with {len(sequence_tensors)} input cached_server_sessions and {len(span_ranges)} peers")

                        sequence_data = self.run_inference_with_tensors(
                            self.input_data, 
                            peers=span_ranges,
                            input_tensor=sequence_tensors
                        )
                        self.validate_inference_results(peer, sequence_data)
                        break #testing

                # print('\naccountant_data\n')
                # pprint.pprint(self.accountant_data.data)

                logger.info("Completed inference validation sequence")

            except Exception as e:
                logger.error(e, exc_info=True)
            finally:
                # # Reset previous epochs cached inference sequence
                # self.cached_inference_sequence = None
                # Remove strict blocks if they are strict
                self.server.remove_strict_block_indices()
                self.server.is_validator = False
                seconds_remaining_in_epoch = self._get_seconds_remaining_in_epoch()
                logger.info(f"Next epoch is {seconds_remaining_in_epoch} seconds away, sleeping for remaining time")
                time.sleep(10)
                continue

    def _is_module_container_healthy(self) -> bool:
        logger.info(f"Verifying validator blocks have begun")
        if self.server.module_container is None:
            # Restart loop
            logger.info("Module container not loaded yet")
            return False

        # Ensure that module container is initialized
        logger.info(f"Verifying validator blocks are healthy")
        if not self.server.module_container.is_healthy():
            logger.info("Module container not healthy yet")
            return False

        return True

    def listen_for_proposals(self):
        # TODO: Implement logic to listen for proposals and process them
        while True:
            time.sleep(300)

            # Get proposal
            proposal = []

            proposal_data = []

            # Check type 
            proposal_type = 0

            # validate data for type of proposal
            # self.validate_proposal(proposal_type, proposal_data)
        
    def run_in_background(self, await_ready=True, timeout=None):
        self.start()

    def shutdown(self):
        logger.info("Shutting down inference validator")
        self.stop.set()
        self.exit()

    def set_deterministic(self):
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def get_peers_validation_spans(self) -> List:
        """
        Get 1 block span for each peer in the subnet: e.g. 0:1, 119:120

        This data is used to check each peer peer by peer and get the min and max blocks to validate
        """
        peers_data_to_validate = self.peers_data_to_validate
        validation_sequences = []
        while len(peers_data_to_validate) > 0:
            validation_sequence = []
            block = 0
            while block < self.num_model_blocks:
                peers = [i for i in peers_data_to_validate if i['span_start'] <= block <= i["span_end"]]
                if len(peers) == 0:
                    block += 1
                    continue

                for peer in peers:
                    validation_sequence.append(
                        {
                            "peer_id": peer['peer_id'],
                            "start": block,
                            "end": block+1,
                        }
                    )
                    peers_data_to_validate.remove(peer)

                block += 1

                if len(peers_data_to_validate) == 0:
                    break
            validation_sequences.append(validation_sequence)
        return validation_sequences

    def get_min_max_start_block_from_sequences(self, sequences) -> List:
        """Get span of blocks validator must get outputs for as Accountant"""
        start_block = 0
        end_block = self.num_model_blocks

        for sequence in sequences:
            start_block = min(sequence, key=lambda x: x['start'])['start']
            end_block = max(sequence, key=lambda x: x['end'])['end']

        return [0, end_block]

    def run_inference_as_accountant(
        self, 
        input_data, 
        peers: List[Dict],
    ):
        try:
            """Run inference and return the results from the span_start to the span_end"""
            # TODO: Get inference only up to the end block required to run inference checks to save on compute
            _input_data = self.tokenizer(input_data, return_tensors="pt")["input_ids"]

            inference_session_data, outputs = self.model.generate_tensors(
                _input_data, 
                peers=peers,
                max_new_tokens=5,
            )

            print("run_inference_as_accountant outputs decode", self.tokenizer.decode(outputs[0]))

            my_inference_sequence_cache = self.get_accountant_inference_results(inference_session_data)
            self.push_inference_sequence_cache(my_inference_sequence_cache)

        except Exception as e:
            logger.warning(f"Inference Validation Error: {e}", exc_info=True)

    def run_inference_with_tensors(
        self, 
        input_data, 
        peers: List[Dict],
        input_tensor: Optional[torch.Tensor] = None, 
    ):
        try:
            """Run inference and return the results from the span_start to the span_end"""
            # TODO: Get inference only up to the end block required to run inference checks to save on compute
            _input_data = self.tokenizer(input_data, return_tensors="pt")["input_ids"]

            inference_session_data, outputs = self.model.generate_tensors(
                _input_data, 
                peers=peers,
                max_new_tokens=5,
                cached_server_sessions=input_tensor
            )

            # print("run_inference_with_tensors outputs decode", self.tokenizer.decode(outputs[0]))
            return inference_session_data
        except Exception as e:
            logger.warning(f"Inference Validation Error: {e}", exc_info=True)

    def push_inference_sequence_cache(self, sequence: List):
        """This data sent in here should only be matched with self.my_peer_id"""
        if self.cached_inference_sequence is None:
            logger.info("push_inference_sequence_cache is None")
            # Push new data if it doesn't already exist
            self.cached_inference_sequence = sequence
        else:
            logger.info("push_inference_sequence_cache is existing")
            for data in sequence:
                span_found = next((x for x in self.cached_inference_sequence if x['server_idx'] == data["server_idx"]), None)
                
                """Append span data if none exists"""
                if span_found is None:
                    self.cached_inference_sequence.append(data)

    def get_account_input_tensors(self, start, end) -> List:
        """Return all sequence outputs that match the start and end blocks"""
        print(f"get_account_input_tensors start {start}, end {end}")

        inference_sequence_cache = [i for i in self.cached_inference_sequence if i['span_start'] == start and i['span_end'] == end]

        if inference_sequence_cache is None or len(inference_sequence_cache) == 0:
            return None

        return inference_sequence_cache

    def validate_inference_results(self, peer_data, inference_session_data):
        peer_id = peer_data["peer_id"]
        start = peer_data["start"]
        end = peer_data["end"]

        peer_validation_data = PeerValidationData(
            input_tensor=None,
            a_tol=ATOL,
            r_tol=RTOL,
            data=[] # PeerInferenceResults
        )
        
        peer_inference_results = PeerInferenceResults(
            span_start=start,
            span_end=end,
            data=[] # PeerInferenceSequenceData
        )

        """Iterate inference results for a given peer"""
        for session in inference_session_data:
            if session["peer_id"] != peer_id:
                continue

            # Get cached accountant inference session data to check against
            span_start = session["span_start"]
            span_end = session["span_end"]
            position = session["position"]

            # Find cached results to compare
            accountant_inference_cache = self.get_inference_by_position(
                self.my_peer_id, 
                self.cached_inference_sequence, 
                span_start, 
                span_end, 
                position
            )
            
            if accountant_inference_cache is None or len(accountant_inference_cache) == 0:
                continue

            # Peers outputs
            outputs = session["outputs"]
            # Accountants outputs
            expected_outputs = accountant_inference_cache["outputs"]

            expected_outputs_tensor_sum = torch.sum(expected_outputs)
            outputs_tensor_sum = torch.sum(outputs)

            tensor_diff = expected_outputs_tensor_sum - outputs_tensor_sum

            valid = torch.allclose(expected_outputs, outputs, rtol=RTOL, atol=ATOL, equal_nan=False)

            logger.info(f"Tensor sum diff is:              {tensor_diff}/{-tensor_diff}")
            logger.info(f"Max tensor sum diff is:          {-ATOL}/{ATOL}")
            logger.info(f"Expected output tensor sum is:   {expected_outputs_tensor_sum}")
            logger.info(f"Validating output tensor sum is: {outputs_tensor_sum}")
            logger.info(f"Inference valid status:          {valid}")

            peer_inference_sequence_data = PeerInferenceSequenceData(
                position=position,
                accountant_tensor_sum=expected_outputs_tensor_sum,
                tensor_sum=outputs_tensor_sum,
                valid=valid
            )

            peer_inference_results.data.append(peer_inference_sequence_data)

        peer_validation_data.data.append(peer_inference_results)

        valid_all = True
        valid = []

        for data in peer_inference_results.data:
            valid.append(data.valid)

        valid_count = len(valid)
        valid_true = sum(valid)

        valid_rate = valid_true / valid_count if valid_count > 0 else 0
        if valid_rate < VTOL:
            valid_all = False

        self.accountant_data.add_data(
            AccountantDataPeerParams(
                peer_id=peer_id,
                valid=valid_all,
                data=peer_validation_data,
            )
        )

    def get_accountant_inference_results(self, inference_session_data) -> List:
        """Append the inference results by the accountant only"""
        inference_data = []
        for data in inference_session_data:
            if data['peer_id'].__eq__(self.my_peer_id):
                inference_data.append(data)
        return inference_data

    def get_inference_by_position(self, peer_id, sequence_data, start, end, position) -> List:
        """Return cached inference sequence data for a given start, end, and position"""
        for data in sequence_data:
            if data['peer_id'] == peer_id and data['span_start'] == start and data['span_end'] == end and data['position'] == position:
                return data
        return None

    def initiate_dishonesty(self, peer_id):
        """
        Propose the peer as dishonest on the blockchain
            If already proposed, then vote
        """

        # Check if proposal already exists
        proposal_exists = True 
        # if proposal_exists:
        #     tx_hash = vote_model_peer_dishonest(
        #         self.client.substrate_interface,
        #         self.client.keypair,
        #         model_id=0,  # Example: model id
        #         peer_id=peer_id,  # Example: peer id to vote as dishonest
        #     )
        # else:
        #     tx_hash = propose_model_peer_dishonest(
        #         self.client.substrate_interface,
        #         self.client.keypair,
        #         model_id=0,  # Example: model id
        #         peer_id=peer_id,  # Example: peer id to vote as dishonest
        #     )

        tx_hash = 0

        print(f"Proposed dishonest peer {peer_id} with transaction hash: {tx_hash}")

    def submit_accountant_data(self, data: str):
        """
        Submit data to the blockchain if chosen accountant on the epoch.

        The data must be formatted as a string to send to the blockchain
        This data can then be used for other subnet nodes to pull from the blockchain storage
        """
        # tx_hash = submit_data(
        #     self.client.substrate_interface,
        #     self.client.keypair,
        #     data=data,
        # )

    def update_peers(self):
        self.peers_data = []
        self.peers_data_to_validate = []
        peers_data_list = get_peers_data_list(self.authorizer)
        if peers_data_list is None or len(peers_data_list) == 0:
            return 
        for peer in peers_data_list:
            logger.info(f"update_peers peer_id: {peer['peer_id'] }")
            if peer['peer_id'] != self.my_peer_id:  # Exclude the validator from the peer list
                self.peers_data.append(peer)

        print("update_peers self.peers_data", self.peers_data)
        self.peers_data_to_validate = self.peers_data

    def _get_peer_data(self, peer_id):
        return next((x for x in self.peers_data_to_validate if x['peer_id'] == peer_id), None)

    def _get_peers_data_in_range(self, span_start: int, span_end: int) -> List:
        peers_data: List = []
        for peer in self.peers_data_to_validate:
            if peer['peer_id'] != self.my_peer_id and peer['span_start'] == span_start and peer['span_end'] == span_end:  # Exclude the validator from the peer list
                peers_data.append(peer)

        return peers_data 
    
    def _get_peers_data_within_range(self, span_start: int, span_end: int) -> List:
        """
        Get peers that are within the given range
        ex: If span_start: 0, span_end: 20
            results = [0:20, 5:15, 10:20]
        """
        peers_data: List = []
        for peer in self.peers_data_to_validate:
            if (peer['peer_id'] != self.my_peer_id and 
                peer['span_start'] >= span_start and 
                peer['span_end'] <= span_end
            ):  # Exclude the validator from the peer list
                peers_data.append(peer)

        return peers_data
    
    def _get_sequence_for_inference(self, peers_data) -> List:
        """"""
        peers = []
        total_peers = len(peers_data)
        min_span = self.num_blocks + 1
        for peer in peers_data:
            span_len = int(peer["span_end"]) - int(peer["span_start"])
            if span_len < min_span:
                min_span = span_len

        max_span = 0

    def _is_accountant(self) -> bool:
        """Check if is already an accountant classification"""
        # accountant_account_id = get_epoch_accountant(
        #     SubstrateConfigCustom.interface, 
        #     SubstrateConfigCustom.keypair,
        #     self.model_id
        # )
        # if accountant_account_id == SubstrateConfigCustom.hotkey:
        #     self.is_accountant = True
        # else:
        #     self.is_accountant = False

        # return self.is_accountant
        return True

    def _is_chosen_accountant(self) -> bool:
        """Check if chosen accountant on epoch"""
        return True
    
    def _get_epoch(self):
        """Do math to get epoch number from blockchain"""
        # block_hash = SubstrateConfigCustom.interface.get_block_hash()
        # block_number = SubstrateConfigCustom.interface.get_block_number(block_hash)
        # network_config = load_network_config()
        # min_required_model_consensus_submit_epochs = network_config.min_required_model_consensus_submit_epochs
        # min_required_peer_consensus_submit_epochs = network_config.min_required_peer_consensus_submit_epochs
        # min_model_peers = network_config.min_model_peers
        # epoch_length = network_config.epoch_length

        return 1

    def _get_seconds_remaining_in_epoch(self) -> int:
        """
        Get how much time is left in the epoch until the next epoch
        
        This is used to wait until the next epoch to begin inference validation again
        """
        return 100
    
    def compare_accountant_data():
        """
        Compare the current accountant data to self
        """
        # accountant_data = get_previous_accountant_data(
        #     SubstrateConfigCustom.interface, 
        #     SubstrateConfigCustom.keypair,
        #     self.model_id
        # )